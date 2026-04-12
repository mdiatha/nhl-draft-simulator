import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as cloudfront from 'aws-cdk-lib/aws-cloudfront';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import { Construct } from 'constructs';

export interface IamConstructProps {
  prefix: string;
  githubOrg: string;
  githubRepo: string;
  ecrRepository: ecr.Repository;
  frontendBucket: s3.Bucket;
  cloudfrontDistribution: cloudfront.Distribution;
  ecsCluster: ecs.Cluster;
  ecsTaskRole: iam.Role;
  ecsTaskExecutionRole: iam.Role;
  lambdaFunction: lambda.Function;
}

/**
 * GitHub Actions OIDC federation — short-lived credentials, no stored AWS keys.
 * The role grants exactly what CI/CD needs: ECR push, ECS deploy, S3 frontend
 * sync, CloudFront invalidation, and Lambda code update.
 */
export class IamConstruct extends Construct {
  public readonly githubActionsRole: iam.Role;

  constructor(scope: Construct, id: string, props: IamConstructProps) {
    super(scope, id);
    const {
      prefix,
      githubOrg,
      githubRepo,
      ecrRepository,
      frontendBucket,
      cloudfrontDistribution,
      ecsCluster,
      ecsTaskRole,
      ecsTaskExecutionRole,
      lambdaFunction,
    } = props;

    // OIDC provider for GitHub Actions
    const oidcProvider = new iam.OpenIdConnectProvider(this, 'GithubOidc', {
      url: 'https://token.actions.githubusercontent.com',
      clientIds: ['sts.amazonaws.com'],
      thumbprints: ['6938fd4d98bab03faadb97b34396831e3780aea1'],
    });

    const repoCondition = githubOrg && githubRepo
      ? `repo:${githubOrg}/${githubRepo}:*`
      : 'repo:*/*:*';

    this.githubActionsRole = new iam.Role(this, 'GithubActionsRole', {
      roleName: `${prefix}-github-actions`,
      assumedBy: new iam.WebIdentityPrincipal(oidcProvider.openIdConnectProviderArn, {
        StringLike: {
          'token.actions.githubusercontent.com:sub': repoCondition,
        },
        StringEquals: {
          'token.actions.githubusercontent.com:aud': 'sts.amazonaws.com',
        },
      }),
    });

    this.githubActionsRole.addToPolicy(
      new iam.PolicyStatement({
        sid: 'ECRAuth',
        actions: ['ecr:GetAuthorizationToken'],
        resources: ['*'],
      }),
    );
    this.githubActionsRole.addToPolicy(
      new iam.PolicyStatement({
        sid: 'ECRPush',
        actions: [
          'ecr:BatchCheckLayerAvailability',
          'ecr:GetDownloadUrlForLayer',
          'ecr:BatchGetImage',
          'ecr:InitiateLayerUpload',
          'ecr:UploadLayerPart',
          'ecr:CompleteLayerUpload',
          'ecr:PutImage',
        ],
        resources: [ecrRepository.repositoryArn],
      }),
    );
    this.githubActionsRole.addToPolicy(
      new iam.PolicyStatement({
        sid: 'FrontendDeploy',
        actions: ['s3:PutObject', 's3:GetObject', 's3:DeleteObject', 's3:ListBucket'],
        resources: [frontendBucket.bucketArn, `${frontendBucket.bucketArn}/*`],
      }),
    );
    this.githubActionsRole.addToPolicy(
      new iam.PolicyStatement({
        sid: 'CloudFrontInvalidate',
        actions: ['cloudfront:CreateInvalidation'],
        resources: [
          `arn:aws:cloudfront::${cdk.Stack.of(this).account}:distribution/${cloudfrontDistribution.distributionId}`,
        ],
      }),
    );
    const region = cdk.Stack.of(this).region;
    const account = cdk.Stack.of(this).account;
    const clusterName = ecsCluster.clusterName;

    this.githubActionsRole.addToPolicy(
      new iam.PolicyStatement({
        sid: 'ECSDescribe',
        actions: ['ecs:DescribeServices', 'ecs:DescribeTasks'],
        resources: [
          `arn:aws:ecs:${region}:${account}:cluster/${clusterName}`,
          `arn:aws:ecs:${region}:${account}:service/${clusterName}/*`,
          `arn:aws:ecs:${region}:${account}:task/${clusterName}/*`,
        ],
      }),
    );
    this.githubActionsRole.addToPolicy(
      new iam.PolicyStatement({
        sid: 'ECSRunTask',
        actions: ['ecs:RunTask'],
        resources: [
          `arn:aws:ecs:${region}:${account}:task-definition/*`,
        ],
        conditions: {
          ArnEquals: {
            'ecs:cluster': `arn:aws:ecs:${region}:${account}:cluster/${clusterName}`,
          },
        },
      }),
    );
    this.githubActionsRole.addToPolicy(
      new iam.PolicyStatement({
        sid: 'ECSUpdateService',
        actions: ['ecs:UpdateService'],
        resources: [
          `arn:aws:ecs:${region}:${account}:service/${clusterName}/*`,
        ],
      }),
    );
    this.githubActionsRole.addToPolicy(
      new iam.PolicyStatement({
        sid: 'LambdaUpdate',
        actions: ['lambda:UpdateFunctionCode'],
        resources: [lambdaFunction.functionArn],
      }),
    );
    this.githubActionsRole.addToPolicy(
      new iam.PolicyStatement({
        sid: 'PassEcsTaskRoles',
        actions: ['iam:PassRole'],
        resources: [ecsTaskRole.roleArn, ecsTaskExecutionRole.roleArn],
      }),
    );
    // CDK deploy permissions — allows GitHub Actions to call cdk deploy
    this.githubActionsRole.addToPolicy(
      new iam.PolicyStatement({
        sid: 'CdkDeploy',
        actions: [
          'cloudformation:DescribeStacks',
          'cloudformation:CreateStack',
          'cloudformation:UpdateStack',
          'cloudformation:DeleteStack',
          'cloudformation:DescribeStackEvents',
          'cloudformation:GetTemplate',
          'cloudformation:ValidateTemplate',
          'cloudformation:CreateChangeSet',
          'cloudformation:DescribeChangeSet',
          'cloudformation:ExecuteChangeSet',
          'cloudformation:DeleteChangeSet',
          'sts:AssumeRole',
        ],
        resources: ['*'],
      }),
    );
  }
}
