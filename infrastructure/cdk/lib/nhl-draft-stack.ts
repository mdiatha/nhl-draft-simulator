import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import { NetworkConstruct } from './constructs/network';
import { StorageConstruct } from './constructs/storage';
import { SecretsConstruct } from './constructs/secrets';
import { ComputeConstruct } from './constructs/compute';
import { ApiConstruct } from './constructs/api';
import { FrontendConstruct } from './constructs/frontend';
import { IngestionConstruct } from './constructs/ingestion';
import { ObservabilityConstruct } from './constructs/observability';
import { IamConstruct } from './constructs/iam';
import { DnsConstruct } from './constructs/dns';
import { BudgetConstruct } from './constructs/budget';

export interface NhlDraftStackProps extends cdk.StackProps {
  // Passed via cdk.json context or --context flags
}

export class NhlDraftStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: NhlDraftStackProps) {
    super(scope, id, props);

    const ctx = this.node.tryGetContext;
    const environment: string = ctx('environment') ?? 'production';
    const prefix = `nhl-draft-${environment}`;

    // ── Context-driven config (set via cdk.json or --context flags) ────────────
    const alertEmail: string = ctx('alertEmail') ?? 'alerts@example.com';
    const githubOrg: string = ctx('githubOrg') ?? '';
    const githubRepo: string = ctx('githubRepo') ?? '';
    const route53ZoneName: string = ctx('route53ZoneName') ?? '';
    const frontendDomainName: string = ctx('frontendDomainName') ?? '';
    const apiDomainName: string = ctx('apiDomainName') ?? '';
    const apiGatewayThrottleBurstLimit: number = Number(ctx('apiGatewayThrottleBurstLimit') ?? 120);
    const apiGatewayThrottleRateLimit: number = Number(ctx('apiGatewayThrottleRateLimit') ?? 60);
    const monthlyBudgetLimitUsd: number = Number(ctx('monthlyBudgetLimitUsd') ?? 40);
    const monthlyBudgetAlertThresholdPct: number = Number(ctx('monthlyBudgetAlertThresholdPct') ?? 80);

    // ── Network ────────────────────────────────────────────────────────────────
    const network = new NetworkConstruct(this, 'Network', { prefix });

    // ── Storage (S3 buckets) ───────────────────────────────────────────────────
    const storage = new StorageConstruct(this, 'Storage', { prefix });

    // ── Secrets & SSM parameters ───────────────────────────────────────────────
    const secrets = new SecretsConstruct(this, 'Secrets', { prefix });

    // ── Compute (ECS on EC2 + RDS) ─────────────────────────────────────────────
    const compute = new ComputeConstruct(this, 'Compute', {
      prefix,
      vpc: network.vpc,
      albSecurityGroup: network.albSg,
      ecsSecurityGroup: network.ecsSg,
      rdsSecurityGroup: network.rdsSg,
      modelsBucket: storage.modelsBucket,
      appSecret: secrets.appSecret,
      ssmParams: secrets.ssmParams,
    });

    // ── API Gateway (HTTP API → VPC Link → ALB) ────────────────────────────────
    const api = new ApiConstruct(this, 'Api', {
      prefix,
      vpc: network.vpc,
      apigwSecurityGroup: network.apigwVpcLinkSg,
      isolatedSubnets: network.isolatedSubnets,
      albListener: compute.albListener,
      throttleBurstLimit: apiGatewayThrottleBurstLimit,
      throttleRateLimit: apiGatewayThrottleRateLimit,
    });

    // ── Ingestion Lambda (daily EventBridge trigger) ───────────────────────────
    const ingestion = new IngestionConstruct(this, 'Ingestion', {
      prefix,
      apiEndpoint: api.httpApi.apiEndpoint,
    });

    // ── Frontend (CloudFront + S3) ─────────────────────────────────────────────
    const frontend = new FrontendConstruct(this, 'Frontend', {
      prefix,
      frontendBucket: storage.frontendBucket,
      frontendDomainName: frontendDomainName || undefined,
      route53ZoneName: route53ZoneName || undefined,
    });

    // ── IAM (GitHub Actions OIDC role) ─────────────────────────────────────────
    const iam = new IamConstruct(this, 'Iam', {
      prefix,
      githubOrg,
      githubRepo,
      ecrRepository: compute.ecrRepository,
      frontendBucket: storage.frontendBucket,
      cloudfrontDistribution: frontend.distribution,
      ecsCluster: compute.ecsCluster,
      ecsTaskRole: compute.ecsTaskRole,
      ecsTaskExecutionRole: compute.ecsTaskExecutionRole,
      lambdaFunction: ingestion.lambdaFunction,
    });

    // ── Observability (CloudWatch alarms, dashboards, SNS) ─────────────────────
    const observability = new ObservabilityConstruct(this, 'Observability', {
      prefix,
      environment,
      alertEmail,
      ecsCluster: compute.ecsCluster,
      ecsService: compute.ecsService,
      alb: compute.alb,
      albTargetGroup: compute.albTargetGroup,
      httpApi: api.httpApi,
      httpApiStage: api.httpApiStage,
      rdsInstance: compute.rdsInstance,
      lambdaFunction: ingestion.lambdaFunction,
    });

    // ── DNS (optional Route 53 custom domains + ACM) ───────────────────────────
    if (route53ZoneName) {
      new DnsConstruct(this, 'Dns', {
        prefix,
        route53ZoneName,
        frontendDomainName: frontendDomainName || undefined,
        apiDomainName: apiDomainName || undefined,
        cloudfrontDistribution: frontend.distribution,
        httpApi: api.httpApi,
        httpApiStage: api.httpApiStage,
      });
    }

    // ── Budget alert ───────────────────────────────────────────────────────────
    new BudgetConstruct(this, 'Budget', {
      prefix,
      limitUsd: monthlyBudgetLimitUsd,
      alertThresholdPct: monthlyBudgetAlertThresholdPct,
      alertEmail,
    });

    // ── Stack outputs ──────────────────────────────────────────────────────────
    new cdk.CfnOutput(this, 'ApiUrl', {
      description: 'FastAPI backend URL via the public API entrypoint',
      value: api.httpApi.apiEndpoint,
    });
    new cdk.CfnOutput(this, 'FrontendUrl', {
      description: 'Frontend HTTPS URL for the React SPA',
      value: `https://${frontend.distribution.distributionDomainName}`,
    });
    new cdk.CfnOutput(this, 'EcrRepositoryUrl', {
      description: 'ECR repository URL for Docker image pushes',
      value: compute.ecrRepository.repositoryUri,
    });
    new cdk.CfnOutput(this, 'EcsClusterName', {
      description: 'ECS cluster name for the backend service',
      value: compute.ecsCluster.clusterName,
    });
    new cdk.CfnOutput(this, 'EcsServiceName', {
      description: 'ECS service name for the backend API',
      value: compute.ecsService.serviceName,
    });
    new cdk.CfnOutput(this, 'EcsMigrationTaskFamily', {
      description: 'ECS task definition family for one-off Alembic migrations',
      value: compute.migrationTaskDefinition.family,
    });
    new cdk.CfnOutput(this, 'ModelBucket', {
      description: 'S3 bucket for ML model artifacts',
      value: storage.modelsBucket.bucketName,
    });
    new cdk.CfnOutput(this, 'FrontendBucket', {
      description: 'S3 bucket — deploy React build here',
      value: storage.frontendBucket.bucketName,
    });
    new cdk.CfnOutput(this, 'CloudfrontDistributionId', {
      description: 'CloudFront distribution ID — needed for cache invalidation',
      value: frontend.distribution.distributionId,
    });
    new cdk.CfnOutput(this, 'GithubActionsRoleArn', {
      description: 'IAM role ARN for GitHub Actions OIDC — set as AWS_DEPLOY_ROLE_ARN secret',
      value: iam.githubActionsRole.roleArn,
    });
    new cdk.CfnOutput(this, 'LambdaFunctionName', {
      description: 'Ingestion Lambda function name',
      value: ingestion.lambdaFunction.functionName,
    });
    new cdk.CfnOutput(this, 'RdsEndpoint', {
      description: 'RDS hostname — used in DATABASE_URL',
      value: compute.rdsInstance.dbInstanceEndpointAddress,
    });
  }
}
