import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as cloudfront from 'aws-cdk-lib/aws-cloudfront';
import * as cloudfront_origins from 'aws-cdk-lib/aws-cloudfront-origins';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as ssm from 'aws-cdk-lib/aws-ssm';
import * as budgets from 'aws-cdk-lib/aws-budgets';
import * as sns from 'aws-cdk-lib/aws-sns';
import * as sns_subscriptions from 'aws-cdk-lib/aws-sns-subscriptions';
import { Construct } from 'constructs';

export class NhlDraftStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const ctx = (key: string) => this.node.tryGetContext(key);
    const environment: string = ctx('environment') ?? 'production';
    const prefix = `nhl-draft-${environment}`;
    const alertEmail: string = ctx('alertEmail') ?? 'alerts@example.com';
    const githubOrg: string = ctx('githubOrg') ?? '';
    const githubRepo: string = ctx('githubRepo') ?? '';
    const monthlyBudgetLimitUsd: number = Number(ctx('monthlyBudgetLimitUsd') ?? 10);
    const monthlyBudgetAlertThresholdPct: number = Number(ctx('monthlyBudgetAlertThresholdPct') ?? 80);

    // ---- ECR repository -------------------------------------------------------
    const ecrRepo = new ecr.Repository(this, 'EcrRepo', {
      repositoryName: 'nhl-draft-api',
      imageScanOnPush: true,
      imageTagMutability: ecr.TagMutability.MUTABLE,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      lifecycleRules: [{ maxImageCount: 5 }],
    });

    // ---- SSM parameters (values set manually after first deploy) --------------
    const makeParam = (logicalId: string, name: string, description: string, contextKey: string) => {
      const value: string = this.node.tryGetContext(contextKey) ?? 'placeholder';
      return new ssm.StringParameter(this, logicalId, {
        parameterName: name,
        description,
        stringValue: value,
        tier: ssm.ParameterTier.STANDARD,
      });
    };
    const anthropicParam = makeParam('AnthropicApiKey', '/nhl-draft/anthropic_api_key', 'Anthropic API key', 'anthropicApiKey');
    const secretKeyParam = makeParam('SecretKey', '/nhl-draft/secret_key', 'FastAPI secret key', 'secretKey');
    const adminApiKeyParam = makeParam('AdminApiKey', '/nhl-draft/admin_api_key', 'Admin API key', 'adminApiKey');
    const dbPasswordParam = makeParam('DbPassword', '/nhl-draft/db_password', 'PostgreSQL password', 'dbPassword');

    // ---- IAM role for the EC2 instance ----------------------------------------
    const instanceRole = new iam.Role(this, 'InstanceRole', {
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSSMManagedInstanceCore'),
      ],
    });
    ecrRepo.grantPull(instanceRole);
    anthropicParam.grantRead(instanceRole);
    secretKeyParam.grantRead(instanceRole);
    adminApiKeyParam.grantRead(instanceRole);
    dbPasswordParam.grantRead(instanceRole);

    // ---- VPC (default VPC — looked up at deploy time via context) ----------------
    // fromLookup requires account+region; in CI without credentials we use a dummy
    // VPC attribute set so cdk synth succeeds for validation.
    const vpc = ec2.Vpc.fromVpcAttributes(this, 'DefaultVpc', {
      vpcId: this.node.tryGetContext('vpcId') ?? 'vpc-00000000',
      availabilityZones: [this.region + 'a', this.region + 'b'],
      publicSubnetIds: [
        this.node.tryGetContext('subnetId1') ?? 'subnet-00000001',
        this.node.tryGetContext('subnetId2') ?? 'subnet-00000002',
      ],
    });

    // ---- Security group -------------------------------------------------------
    const sg = new ec2.SecurityGroup(this, 'InstanceSg', {
      vpc,
      description: 'NHL Draft EC2 - allow HTTP/HTTPS inbound',
      allowAllOutbound: true,
    });
    sg.addIngressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(80), 'HTTP');
    sg.addIngressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(443), 'HTTPS');

    // ---- User-data bootstrap script -------------------------------------------
    const userData = ec2.UserData.forLinux();
    userData.addCommands(
      '#!/bin/bash',
      'set -euxo pipefail',
      '',
      '# Install Docker',
      'dnf update -y',
      'dnf install -y docker',
      'systemctl enable --now docker',
      'usermod -aG docker ec2-user',
      '',
      '# Install Docker Compose plugin',
      'mkdir -p /usr/local/lib/docker/cli-plugins',
      'curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 -o /usr/local/lib/docker/cli-plugins/docker-compose',
      'chmod +x /usr/local/lib/docker/cli-plugins/docker-compose',
      '',
      '# Install AWS CLI v2',
      'dnf install -y awscli',
      '',
      '# Write .env from SSM at boot',
      `REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)`,
      `get_param() { aws ssm get-parameter --name "$1" --with-decryption --query Parameter.Value --output text --region $REGION; }`,
      `ECR_ACCOUNT=$(aws sts get-caller-identity --query Account --output text --region $REGION)`,
      `ECR_REGISTRY="$ECR_ACCOUNT.dkr.ecr.$REGION.amazonaws.com"`,
      '',
      'cat > /home/ec2-user/.env <<EOF',
      `APP_ENV=production`,
      `LOG_LEVEL=INFO`,
      `NHL_API_BASE_URL=https://api-web.nhle.com/v1`,
      `ANTHROPIC_MODEL=claude-haiku-4-5-20251001`,
      `OLLAMA_BASE_URL=http://ollama:11434`,
      `DATABASE_URL=postgresql+psycopg2://nhl:$(get_param /nhl-draft/db_password)@db:5432/nhl_draft`,
      `REDIS_URL=redis://redis:6379/0`,
      `SECRET_KEY=$(get_param /nhl-draft/secret_key)`,
      `ADMIN_API_KEY=$(get_param /nhl-draft/admin_api_key)`,
      `ANTHROPIC_API_KEY=$(get_param /nhl-draft/anthropic_api_key)`,
      'EOF',
      'chown ec2-user:ec2-user /home/ec2-user/.env',
      '',
      '# Write docker-compose.prod.yml',
      `aws ssm get-parameter --name /nhl-draft/compose --query Parameter.Value --output text --region $REGION > /home/ec2-user/docker-compose.yml || true`,
      '',
      '# Login to ECR and start services',
      `aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_REGISTRY`,
      `cd /home/ec2-user`,
      `docker compose up -d --pull always 2>&1 | tee /var/log/docker-compose-boot.log`,
    );

    // ---- EC2 instance ---------------------------------------------------------
    const instance = new ec2.Instance(this, 'Instance', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.SMALL),
      machineImage: ec2.MachineImage.latestAmazonLinux2023(),
      vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
      securityGroup: sg,
      role: instanceRole,
      userData,
      blockDevices: [{
        deviceName: '/dev/xvda',
        volume: ec2.BlockDeviceVolume.ebs(30, {
          volumeType: ec2.EbsDeviceVolumeType.GP3,
          encrypted: true,
        }),
      }],
    });
    cdk.Tags.of(instance).add('Name', `${prefix}-server`);

    // ---- Elastic IP ----------------------------------------------------------
    const eip = new ec2.CfnEIP(this, 'ElasticIp', {
      instanceId: instance.instanceId,
      tags: [{ key: 'Name', value: `${prefix}-eip` }],
    });

    // ---- S3 frontend bucket --------------------------------------------------
    const frontendBucket = new s3.Bucket(this, 'FrontendBucket', {
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });
    cdk.Tags.of(frontendBucket).add('Name', `${prefix}-frontend`);

    // ---- CloudFront ----------------------------------------------------------
    // CloudFront requires a domain name (not raw IP) as origin.
    // The EC2 DNS is derived from the Elastic IP (54.174.178.24 -> ec2-54-174-178-24.compute-1.amazonaws.com).
    // If the EIP changes, update this value accordingly.
    const ec2ApiDns: string = ctx('ec2ApiDns') ?? 'ec2-54-174-178-24.compute-1.amazonaws.com';
    const apiOrigin = new cloudfront_origins.HttpOrigin(ec2ApiDns, {
      protocolPolicy: cloudfront.OriginProtocolPolicy.HTTP_ONLY,
      httpPort: 80,
      connectionAttempts: 2,
      connectionTimeout: cdk.Duration.seconds(5),
      readTimeout: cdk.Duration.seconds(60),
    });

    const distribution = new cloudfront.Distribution(this, 'Distribution', {
      comment: 'NHL Draft Simulator - React SPA',
      defaultRootObject: 'index.html',
      defaultBehavior: {
        origin: cloudfront_origins.S3BucketOrigin.withOriginAccessControl(frontendBucket),
        allowedMethods: cloudfront.AllowedMethods.ALLOW_GET_HEAD,
        viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
        compress: true,
        cachePolicy: cloudfront.CachePolicy.CACHING_OPTIMIZED,
      },
      additionalBehaviors: {
        '/api/*': {
          origin: apiOrigin,
          allowedMethods: cloudfront.AllowedMethods.ALLOW_ALL,
          viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
          cachePolicy: cloudfront.CachePolicy.CACHING_DISABLED,
          originRequestPolicy: cloudfront.OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER,
        },
      },
      errorResponses: [{
        // S3 returns 403 for missing keys (not 404). Rewrite to index.html for SPA routing.
        // We intentionally do NOT rewrite 404 so API 404s (from EC2) pass through unchanged.
        httpStatus: 403,
        responseHttpStatus: 200,
        responsePagePath: '/index.html',
        ttl: cdk.Duration.seconds(0),
      }],
      priceClass: cloudfront.PriceClass.PRICE_CLASS_100,
    });

    // ---- GitHub Actions OIDC role --------------------------------------------
    const oidcProvider = new iam.OpenIdConnectProvider(this, 'GithubOidc', {
      url: 'https://token.actions.githubusercontent.com',
      clientIds: ['sts.amazonaws.com'],
    });

    const githubActionsRole = new iam.Role(this, 'GithubActionsRole', {
      assumedBy: new iam.WebIdentityPrincipal(oidcProvider.openIdConnectProviderArn, {
        StringEquals: {
          'token.actions.githubusercontent.com:aud': 'sts.amazonaws.com',
        },
        StringLike: {
          'token.actions.githubusercontent.com:sub': `repo:${githubOrg}/${githubRepo}:*`,
        },
      }),
      description: 'Role assumed by GitHub Actions for deployments',
    });
    ecrRepo.grantPullPush(githubActionsRole);
    frontendBucket.grantReadWrite(githubActionsRole);
    githubActionsRole.addToPolicy(new iam.PolicyStatement({
      actions: ['cloudfront:CreateInvalidation'],
      resources: [`arn:aws:cloudfront::${this.account}:distribution/${distribution.distributionId}`],
    }));
    githubActionsRole.addToPolicy(new iam.PolicyStatement({
      actions: ['ssm:SendCommand', 'ssm:GetCommandInvocation'],
      resources: ['*'],
    }));

    // ---- Budget alert --------------------------------------------------------
    const alertTopic = new sns.Topic(this, 'AlertTopic', { displayName: `${prefix}-alerts` });
    alertTopic.addSubscription(new sns_subscriptions.EmailSubscription(alertEmail));

    new budgets.CfnBudget(this, 'MonthlyBudget', {
      budget: {
        budgetName: `${prefix}-monthly`,
        budgetType: 'COST',
        timeUnit: 'MONTHLY',
        budgetLimit: { amount: monthlyBudgetLimitUsd, unit: 'USD' },
      },
      notificationsWithSubscribers: [{
        notification: {
          notificationType: 'ACTUAL',
          comparisonOperator: 'GREATER_THAN',
          threshold: monthlyBudgetAlertThresholdPct,
          thresholdType: 'PERCENTAGE',
        },
        subscribers: [{ subscriptionType: 'EMAIL', address: alertEmail }],
      }],
    });

    // ---- Outputs -------------------------------------------------------------
    new cdk.CfnOutput(this, 'InstancePublicIp', {
      description: 'EC2 public IP - API is at http://<ip>:8000',
      value: eip.attrPublicIp,
    });
    new cdk.CfnOutput(this, 'ApiUrl', {
      description: 'API base URL',
      value: `http://${eip.attrPublicIp}`,
    });
    new cdk.CfnOutput(this, 'FrontendUrl', {
      description: 'Frontend HTTPS URL',
      value: `https://${distribution.distributionDomainName}`,
    });
    new cdk.CfnOutput(this, 'EcrRepositoryUrl', {
      description: 'ECR repository URL',
      value: ecrRepo.repositoryUri,
    });
    new cdk.CfnOutput(this, 'FrontendBucketName', {
      description: 'S3 bucket for frontend',
      value: frontendBucket.bucketName,
    });
    new cdk.CfnOutput(this, 'CloudfrontDistributionId', {
      description: 'CloudFront distribution ID',
      value: distribution.distributionId,
    });
    new cdk.CfnOutput(this, 'GithubActionsRoleArn', {
      description: 'IAM role for GitHub Actions - set as AWS_DEPLOY_ROLE_ARN secret',
      value: githubActionsRole.roleArn,
    });
  }
}
