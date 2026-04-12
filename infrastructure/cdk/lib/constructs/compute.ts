import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import * as elbv2 from 'aws-cdk-lib/aws-elasticloadbalancingv2';
import * as rds from 'aws-cdk-lib/aws-rds';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import * as ssm from 'aws-cdk-lib/aws-ssm';
import * as autoscaling from 'aws-cdk-lib/aws-autoscaling';
import * as logs from 'aws-cdk-lib/aws-logs';
import { Construct } from 'constructs';
import { SsmParams } from './secrets';

export interface ComputeConstructProps {
  prefix: string;
  vpc: ec2.Vpc;
  albSecurityGroup: ec2.SecurityGroup;
  ecsSecurityGroup: ec2.SecurityGroup;
  rdsSecurityGroup: ec2.SecurityGroup;
  modelsBucket: s3.Bucket;
  appSecret: secretsmanager.Secret;
  ssmParams: SsmParams;
}

export class ComputeConstruct extends Construct {
  public readonly ecrRepository: ecr.Repository;
  public readonly ecsCluster: ecs.Cluster;
  public readonly alb: elbv2.ApplicationLoadBalancer;
  public readonly albListener: elbv2.ApplicationListener;
  public readonly albTargetGroup: elbv2.ApplicationTargetGroup;
  public readonly ecsService: ecs.Ec2Service;
  public readonly ecsTaskRole: iam.Role;
  public readonly ecsTaskExecutionRole: iam.Role;
  public readonly apiTaskDefinition: ecs.Ec2TaskDefinition;
  public readonly migrationTaskDefinition: ecs.Ec2TaskDefinition;
  public readonly rdsInstance: rds.DatabaseInstance;

  constructor(scope: Construct, id: string, props: ComputeConstructProps) {
    super(scope, id);
    const {
      prefix,
      vpc,
      albSecurityGroup,
      ecsSecurityGroup,
      rdsSecurityGroup,
      modelsBucket,
      appSecret,
      ssmParams,
    } = props;

    // ── ECR repository ─────────────────────────────────────────────────────────
    this.ecrRepository = new ecr.Repository(this, 'EcrRepo', {
      repositoryName: 'nhl-draft-api',
      imageScanOnPush: true,
      imageTagMutability: ecr.TagMutability.MUTABLE,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      lifecycleRules: [
        {
          rulePriority: 1,
          description: 'Keep last 5 images, expire the rest',
          maxImageCount: 5,
        },
      ],
    });
    cdk.Tags.of(this.ecrRepository).add('Name', `${prefix}-ecr`);

    // ── CloudWatch log groups ──────────────────────────────────────────────────
    const ecsLogGroup = new logs.LogGroup(this, 'EcsLogGroup', {
      logGroupName: `/ecs/${prefix}-api`,
      retention: logs.RetentionDays.ONE_MONTH,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });
    cdk.Tags.of(ecsLogGroup).add('Name', `${prefix}-ecs-api-logs`);

    // ── ECS cluster ────────────────────────────────────────────────────────────
    this.ecsCluster = new ecs.Cluster(this, 'EcsCluster', {
      clusterName: `${prefix}-cluster`,
      vpc,
      containerInsights: true,
    });

    // ── ECS container instance role ────────────────────────────────────────────
    const instanceRole = new iam.Role(this, 'EcsInstanceRole', {
      roleName: `${prefix}-ecs-instance-role`,
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSSMManagedInstanceCore'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AmazonEC2ContainerServiceforEC2Role'),
      ],
    });
    instanceRole.addToPolicy(
      new iam.PolicyStatement({
        sid: 'ECRAuth',
        actions: ['ecr:GetAuthorizationToken'],
        resources: ['*'],
      }),
    );
    instanceRole.addToPolicy(
      new iam.PolicyStatement({
        sid: 'ECRPull',
        actions: [
          'ecr:BatchCheckLayerAvailability',
          'ecr:GetDownloadUrlForLayer',
          'ecr:BatchGetImage',
        ],
        resources: [this.ecrRepository.repositoryArn],
      }),
    );

    const instanceProfile = new iam.CfnInstanceProfile(this, 'EcsInstanceProfile', {
      instanceProfileName: `${prefix}-ecs-profile`,
      roles: [instanceRole.roleName],
    });

    // ── Auto Scaling Group (t3.micro ECS instances) ────────────────────────────
    const ecsAmi = ec2.MachineImage.fromSsmParameter(
      '/aws/service/ecs/optimized-ami/amazon-linux-2/recommended/image_id',
    );

    const userData = ec2.UserData.forLinux();
    userData.addCommands(
      'set -euo pipefail',
      `cat > /etc/ecs/ecs.config << 'ECSCFG'`,
      `ECS_CLUSTER=${this.ecsCluster.clusterName}`,
      'ECS_IMAGE_PULL_BEHAVIOR=always',
      'ECS_ENABLE_TASK_IAM_ROLE=true',
      'ECS_ENABLE_TASK_IAM_ROLE_NETWORK_HOST=true',
      'ECSCFG',
      // Small swap so single t3.micro is less brittle
      'dd if=/dev/zero of=/swapfile bs=128M count=16',
      'chmod 600 /swapfile',
      'mkswap /swapfile',
      'swapon /swapfile',
      "echo '/swapfile swap swap defaults 0 0' >> /etc/fstab",
      'systemctl enable --now ecs',
    );

    const launchTemplate = new ec2.LaunchTemplate(this, 'EcsLaunchTemplate', {
      launchTemplateName: `${prefix}-ecs`,
      machineImage: ecsAmi,
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.MICRO),
      securityGroup: ecsSecurityGroup,
      userData,
      blockDevices: [
        {
          deviceName: '/dev/xvda',
          volume: ec2.BlockDeviceVolume.ebs(20, {
            volumeType: ec2.EbsDeviceVolumeType.GP3,
            encrypted: true,
          }),
        },
      ],
    });
    // Attach instance profile to launch template
    const cfnLt = launchTemplate.node.defaultChild as ec2.CfnLaunchTemplate;
    cfnLt.addPropertyOverride(
      'LaunchTemplateData.IamInstanceProfile.Name',
      instanceProfile.ref,
    );

    const asg = new autoscaling.AutoScalingGroup(this, 'EcsAsg', {
      autoScalingGroupName: `${prefix}-ecs`,
      vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
      minCapacity: 1,
      maxCapacity: 1,
      desiredCapacity: 1,
      launchTemplate,
      healthCheck: autoscaling.HealthCheck.ec2(),
    });
    cdk.Tags.of(asg).add('Name', `${prefix}-ecs-instance`);

    const capacityProvider = new ecs.AsgCapacityProvider(this, 'AsgCapacityProvider', {
      autoScalingGroup: asg,
      enableManagedTerminationProtection: false,
    });
    this.ecsCluster.addAsgCapacityProvider(capacityProvider);

    // ── Internal ALB ───────────────────────────────────────────────────────────
    this.alb = new elbv2.ApplicationLoadBalancer(this, 'Alb', {
      loadBalancerName: `${prefix}-api`,
      vpc,
      internetFacing: false,
      securityGroup: albSecurityGroup,
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_ISOLATED },
      idleTimeout: cdk.Duration.seconds(300),
    });
    cdk.Tags.of(this.alb).add('Name', `${prefix}-api-alb`);

    this.albTargetGroup = new elbv2.ApplicationTargetGroup(this, 'AlbTargetGroup', {
      targetGroupName: `${prefix}-api`,
      vpc,
      port: 8000,
      protocol: elbv2.ApplicationProtocol.HTTP,
      targetType: elbv2.TargetType.INSTANCE,
      healthCheck: {
        enabled: true,
        path: '/health',
        healthyHttpCodes: '200',
        healthyThresholdCount: 2,
        unhealthyThresholdCount: 2,
        interval: cdk.Duration.seconds(30),
        timeout: cdk.Duration.seconds(5),
      },
    });
    cdk.Tags.of(this.albTargetGroup).add('Name', `${prefix}-api-tg`);

    this.albListener = this.alb.addListener('HttpListener', {
      port: 80,
      protocol: elbv2.ApplicationProtocol.HTTP,
      defaultTargetGroups: [this.albTargetGroup],
    });

    // ── RDS PostgreSQL ─────────────────────────────────────────────────────────
    const dbSubnetGroup = new rds.SubnetGroup(this, 'DbSubnetGroup', {
      description: `${prefix} DB subnet group`,
      vpc,
      subnetGroupName: `${prefix}-db-subnet-group`,
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_ISOLATED },
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    const dbCredentials = rds.Credentials.fromGeneratedSecret('nhl', {
      secretName: `${prefix}-db-credentials`,
      excludeCharacters: '/@" ', // avoid chars that break connection strings
    });

    this.rdsInstance = new rds.DatabaseInstance(this, 'RdsInstance', {
      instanceIdentifier: `${prefix}-db`,
      engine: rds.DatabaseInstanceEngine.postgres({
        version: rds.PostgresEngineVersion.VER_16,
      }),
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.MICRO),
      credentials: dbCredentials,
      databaseName: 'nhl_draft',
      storageType: rds.StorageType.GP2,
      allocatedStorage: 20,
      maxAllocatedStorage: 100,
      storageEncrypted: true,
      vpc,
      subnetGroup: dbSubnetGroup,
      securityGroups: [rdsSecurityGroup],
      publiclyAccessible: false,
      multiAz: false, // single-AZ saves ~$15/mo
      backupRetention: cdk.Duration.days(7),
      preferredBackupWindow: '03:00-04:00',
      preferredMaintenanceWindow: 'Mon:04:00-Mon:05:00',
      deletionProtection: true,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      cloudwatchLogsExports: ['postgresql'],
    });
    cdk.Tags.of(this.rdsInstance).add('Name', `${prefix}-db`);

    // ── ECS task execution role ────────────────────────────────────────────────
    this.ecsTaskExecutionRole = new iam.Role(this, 'EcsTaskExecutionRole', {
      roleName: `${prefix}-ecs-task-execution`,
      assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AmazonECSTaskExecutionRolePolicy'),
      ],
    });
    this.ecsTaskExecutionRole.addToPolicy(
      new iam.PolicyStatement({
        actions: ['ssm:GetParameters', 'secretsmanager:GetSecretValue'],
        resources: [
          ssmParams.anthropicApiKey.parameterArn,
          ssmParams.voyageApiKey.parameterArn,
          ssmParams.secretKey.parameterArn,
          ssmParams.adminApiKey.parameterArn,
          appSecret.secretArn,
        ],
      }),
    );

    // ── ECS task role ──────────────────────────────────────────────────────────
    this.ecsTaskRole = new iam.Role(this, 'EcsTaskRole', {
      roleName: `${prefix}-ecs-task`,
      assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
    });
    modelsBucket.grantReadWrite(this.ecsTaskRole);
    appSecret.grantRead(this.ecsTaskRole);
    this.ecsTaskRole.addToPolicy(
      new iam.PolicyStatement({
        sid: 'SSMParameterRead',
        actions: ['ssm:GetParameter', 'ssm:GetParameters', 'ssm:GetParametersByPath'],
        resources: [
          `arn:aws:ssm:${cdk.Stack.of(this).region}:${cdk.Stack.of(this).account}:parameter/nhl-draft/*`,
        ],
      }),
    );
    this.ecsTaskRole.addToPolicy(
      new iam.PolicyStatement({
        sid: 'CloudWatchMetrics',
        actions: ['cloudwatch:PutMetricData'],
        resources: ['*'],
      }),
    );

    // ── Shared environment / secrets for task definitions ──────────────────────
    const containerEnvironment: { [key: string]: string } = {
      APP_ENV: 'production',
      ENVIRONMENT: 'production',
      AWS_REGION: cdk.Stack.of(this).region,
      AWS_SECRETS_NAME: appSecret.secretName,
      AWS_S3_BUCKET: modelsBucket.bucketName,
    };

    const containerSecrets: { [key: string]: ecs.Secret } = {
      ANTHROPIC_API_KEY: ecs.Secret.fromSsmParameter(ssmParams.anthropicApiKey),
      VOYAGE_API_KEY: ecs.Secret.fromSsmParameter(ssmParams.voyageApiKey),
      SECRET_KEY: ecs.Secret.fromSsmParameter(ssmParams.secretKey),
      ADMIN_API_KEY: ecs.Secret.fromSsmParameter(ssmParams.adminApiKey),
    };

    // ── API task definition ────────────────────────────────────────────────────
    this.apiTaskDefinition = new ecs.Ec2TaskDefinition(this, 'ApiTaskDef', {
      family: `${prefix}-api`,
      networkMode: ecs.NetworkMode.BRIDGE,
      executionRole: this.ecsTaskExecutionRole,
      taskRole: this.ecsTaskRole,
    });

    this.apiTaskDefinition.addContainer('api', {
      image: ecs.ContainerImage.fromEcrRepository(this.ecrRepository, 'latest'),
      cpu: 256,
      memoryLimitMiB: 768,
      essential: true,
      portMappings: [{ containerPort: 8000, hostPort: 8000, protocol: ecs.Protocol.TCP }],
      environment: containerEnvironment,
      secrets: containerSecrets,
      logging: ecs.LogDrivers.awsLogs({
        logGroup: ecsLogGroup,
        streamPrefix: 'api',
      }),
      command: [
        'uvicorn',
        'app.main:app',
        '--host', '0.0.0.0',
        '--port', '8000',
        '--timeout-keep-alive', '120',
        '--log-level', 'info',
      ],
    });

    // ── Migration task definition ──────────────────────────────────────────────
    this.migrationTaskDefinition = new ecs.Ec2TaskDefinition(this, 'MigrationTaskDef', {
      family: `${prefix}-api-migrate`,
      networkMode: ecs.NetworkMode.BRIDGE,
      executionRole: this.ecsTaskExecutionRole,
      taskRole: this.ecsTaskRole,
    });

    this.migrationTaskDefinition.addContainer('api', {
      image: ecs.ContainerImage.fromEcrRepository(this.ecrRepository, 'latest'),
      cpu: 256,
      memoryLimitMiB: 512,
      essential: true,
      environment: containerEnvironment,
      secrets: containerSecrets,
      logging: ecs.LogDrivers.awsLogs({
        logGroup: ecsLogGroup,
        streamPrefix: 'migrate',
      }),
      command: ['alembic', 'upgrade', 'head'],
    });

    // ── ECS service ────────────────────────────────────────────────────────────
    this.ecsService = new ecs.Ec2Service(this, 'EcsService', {
      serviceName: `${prefix}-api`,
      cluster: this.ecsCluster,
      taskDefinition: this.apiTaskDefinition,
      desiredCount: 1,
      capacityProviderStrategies: [
        { capacityProvider: capacityProvider.capacityProviderName, weight: 1 },
      ],
      healthCheckGracePeriod: cdk.Duration.seconds(60),
      minHealthyPercent: 0,
      maxHealthyPercent: 100,
    });

    this.ecsService.attachToApplicationTargetGroup(this.albTargetGroup);
  }
}
