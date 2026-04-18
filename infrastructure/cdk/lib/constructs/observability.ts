import * as cdk from 'aws-cdk-lib';
import * as cloudwatch from 'aws-cdk-lib/aws-cloudwatch';
import * as cloudwatch_actions from 'aws-cdk-lib/aws-cloudwatch-actions';
import * as sns from 'aws-cdk-lib/aws-sns';
import * as sns_subscriptions from 'aws-cdk-lib/aws-sns-subscriptions';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as elbv2 from 'aws-cdk-lib/aws-elasticloadbalancingv2';
import * as apigwv2 from 'aws-cdk-lib/aws-apigatewayv2';
import * as rds from 'aws-cdk-lib/aws-rds';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import { Construct } from 'constructs';

export interface ObservabilityConstructProps {
  prefix: string;
  environment: string;
  alertEmail: string;
  ecsCluster: ecs.Cluster;
  ecsService: ecs.Ec2Service;
  alb: elbv2.ApplicationLoadBalancer;
  albTargetGroup: elbv2.ApplicationTargetGroup;
  httpApi: apigwv2.HttpApi;
  httpApiStage: apigwv2.HttpStage;
  rdsInstance: rds.DatabaseInstance;
  lambdaFunction: lambda.Function;
}

export class ObservabilityConstruct extends Construct {
  constructor(scope: Construct, id: string, props: ObservabilityConstructProps) {
    super(scope, id);
    const {
      prefix,
      environment,
      alertEmail,
      ecsCluster,
      ecsService,
      alb,
      albTargetGroup,
      httpApi,
      httpApiStage,
      rdsInstance,
      lambdaFunction,
    } = props;

    // ── SNS: ops alerts → email ────────────────────────────────────────────────
    const opsTopic = new sns.Topic(this, 'OpsAlertsTopic', {
      topicName: `${prefix}-ops-alerts`,
    });
    cdk.Tags.of(opsTopic).add('Name', `${prefix}-ops-alerts`);
    opsTopic.addSubscription(new sns_subscriptions.EmailSubscription(alertEmail));

    const snsAction = new cloudwatch_actions.SnsAction(opsTopic);

    // ── Application log group ──────────────────────────────────────────────────
    const appLogGroup = new logs.LogGroup(this, 'AppLogGroup', {
      logGroupName: '/nhl-draft/application',
      retention: logs.RetentionDays.ONE_MONTH,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });
    cdk.Tags.of(appLogGroup).add('Name', `${prefix}-app-logs`);

    // ── Alarms ─────────────────────────────────────────────────────────────────

    const ecsCpuAlarm = new cloudwatch.Alarm(this, 'EcsCpuHighAlarm', {
      alarmName: `${prefix}-ecs-cpu-high`,
      alarmDescription: 'ECS service CPU > 80% for 10 minutes',
      metric: new cloudwatch.Metric({
        namespace: 'AWS/ECS',
        metricName: 'CPUUtilization',
        dimensionsMap: {
          ClusterName: ecsCluster.clusterName,
          ServiceName: ecsService.serviceName,
        },
        statistic: 'Average',
        period: cdk.Duration.minutes(5),
      }),
      comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
      threshold: 80,
      evaluationPeriods: 2,
      treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
    });
    ecsCpuAlarm.addAlarmAction(snsAction);
    ecsCpuAlarm.addOkAction(snsAction);
    cdk.Tags.of(ecsCpuAlarm).add('Name', `${prefix}-ecs-cpu-alarm`);

    const ecsMemAlarm = new cloudwatch.Alarm(this, 'EcsMemHighAlarm', {
      alarmName: `${prefix}-ecs-memory-high`,
      alarmDescription: 'ECS service memory > 85% for 10 minutes',
      metric: new cloudwatch.Metric({
        namespace: 'AWS/ECS',
        metricName: 'MemoryUtilization',
        dimensionsMap: {
          ClusterName: ecsCluster.clusterName,
          ServiceName: ecsService.serviceName,
        },
        statistic: 'Average',
        period: cdk.Duration.minutes(5),
      }),
      comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
      threshold: 85,
      evaluationPeriods: 2,
      treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
    });
    ecsMemAlarm.addAlarmAction(snsAction);
    cdk.Tags.of(ecsMemAlarm).add('Name', `${prefix}-ecs-memory-alarm`);

    const albUnhealthyAlarm = new cloudwatch.Alarm(this, 'AlbUnhealthyAlarm', {
      alarmName: `${prefix}-alb-unhealthy-hosts`,
      alarmDescription: 'ALB target group has unhealthy backend targets',
      metric: new cloudwatch.Metric({
        namespace: 'AWS/ApplicationELB',
        metricName: 'UnHealthyHostCount',
        dimensionsMap: {
          LoadBalancer: alb.loadBalancerFullName,
          TargetGroup: albTargetGroup.targetGroupFullName,
        },
        statistic: 'Average',
        period: cdk.Duration.minutes(1),
      }),
      comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
      threshold: 0,
      evaluationPeriods: 2,
      treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
    });
    albUnhealthyAlarm.addAlarmAction(snsAction);
    cdk.Tags.of(albUnhealthyAlarm).add('Name', `${prefix}-alb-unhealthy-hosts-alarm`);

    const apigw5xxAlarm = new cloudwatch.Alarm(this, 'ApigwErrorAlarm', {
      alarmName: `${prefix}-api-gateway-5xx`,
      alarmDescription: 'API Gateway returned 5xx responses',
      metric: new cloudwatch.Metric({
        namespace: 'AWS/ApiGateway',
        metricName: '5xx',
        dimensionsMap: {
          ApiId: httpApi.apiId,
          Stage: '$default',
        },
        statistic: 'Sum',
        period: cdk.Duration.minutes(5),
      }),
      comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
      threshold: 0,
      evaluationPeriods: 1,
      treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
    });
    apigw5xxAlarm.addAlarmAction(snsAction);
    cdk.Tags.of(apigw5xxAlarm).add('Name', `${prefix}-api-gateway-5xx-alarm`);

    const apigwLatencyAlarm = new cloudwatch.Alarm(this, 'ApigwLatencyAlarm', {
      alarmName: `${prefix}-api-gateway-latency-high`,
      alarmDescription: 'API Gateway latency > 5 seconds on average',
      metric: new cloudwatch.Metric({
        namespace: 'AWS/ApiGateway',
        metricName: 'Latency',
        dimensionsMap: {
          ApiId: httpApi.apiId,
          Stage: '$default',
        },
        statistic: 'Average',
        period: cdk.Duration.minutes(5),
      }),
      comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
      threshold: 5000,
      evaluationPeriods: 2,
      treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
    });
    apigwLatencyAlarm.addAlarmAction(snsAction);
    cdk.Tags.of(apigwLatencyAlarm).add('Name', `${prefix}-api-gateway-latency-alarm`);

    const rdsCpuAlarm = new cloudwatch.Alarm(this, 'RdsCpuAlarm', {
      alarmName: `${prefix}-rds-cpu-high`,
      alarmDescription: 'RDS CPU > 80% for 10 minutes',
      metric: new cloudwatch.Metric({
        namespace: 'AWS/RDS',
        metricName: 'CPUUtilization',
        dimensionsMap: { DBInstanceIdentifier: rdsInstance.instanceIdentifier },
        statistic: 'Average',
        period: cdk.Duration.minutes(5),
      }),
      comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
      threshold: 80,
      evaluationPeriods: 2,
      treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
    });
    rdsCpuAlarm.addAlarmAction(snsAction);
    cdk.Tags.of(rdsCpuAlarm).add('Name', `${prefix}-rds-cpu-alarm`);

    const lambdaErrorAlarm = new cloudwatch.Alarm(this, 'LambdaErrorAlarm', {
      alarmName: `${prefix}-lambda-ingestion-errors`,
      alarmDescription: 'Ingestion Lambda returned an error - check CloudWatch logs',
      metric: new cloudwatch.Metric({
        namespace: 'AWS/Lambda',
        metricName: 'Errors',
        dimensionsMap: { FunctionName: lambdaFunction.functionName },
        statistic: 'Sum',
        period: cdk.Duration.hours(24),
      }),
      comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
      threshold: 0,
      evaluationPeriods: 1,
      treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
    });
    lambdaErrorAlarm.addAlarmAction(snsAction);
    cdk.Tags.of(lambdaErrorAlarm).add('Name', `${prefix}-lambda-errors-alarm`);

    // SLA monitor: alarms if ingestion hasn't run in 26 hours
    const lambdaSlaAlarm = new cloudwatch.Alarm(this, 'LambdaSlaAlarm', {
      alarmName: `${prefix}-lambda-ingestion-sla`,
      alarmDescription: 'Ingestion Lambda has not run in 26 hours - check EventBridge rule',
      metric: new cloudwatch.Metric({
        namespace: 'AWS/Lambda',
        metricName: 'Invocations',
        dimensionsMap: { FunctionName: lambdaFunction.functionName },
        statistic: 'Sum',
        period: cdk.Duration.seconds(93600), // 26 hours
      }),
      comparisonOperator: cloudwatch.ComparisonOperator.LESS_THAN_THRESHOLD,
      threshold: 1,
      evaluationPeriods: 1,
      treatMissingData: cloudwatch.TreatMissingData.BREACHING, // no data = didn't run = alarm
    });
    lambdaSlaAlarm.addAlarmAction(snsAction);
    cdk.Tags.of(lambdaSlaAlarm).add('Name', `${prefix}-lambda-sla-alarm`);

    // ── Dashboard ──────────────────────────────────────────────────────────────
    new cloudwatch.Dashboard(this, 'Dashboard', {
      dashboardName: 'NHLDraftSimulator',
      widgets: [
        [
          new cloudwatch.TextWidget({
            markdown: `## NHL Draft Simulator - ${environment}`,
            width: 24,
            height: 1,
          }),
        ],
        [
          new cloudwatch.GraphWidget({
            title: 'ECS API - CPU & Memory',
            width: 8,
            height: 6,
            left: [
              new cloudwatch.Metric({
                namespace: 'AWS/ECS',
                metricName: 'CPUUtilization',
                dimensionsMap: {
                  ClusterName: ecsCluster.clusterName,
                  ServiceName: ecsService.serviceName,
                },
                statistic: 'Average',
                label: 'CPU %',
                period: cdk.Duration.minutes(5),
              }),
            ],
            right: [
              new cloudwatch.Metric({
                namespace: 'AWS/ECS',
                metricName: 'MemoryUtilization',
                dimensionsMap: {
                  ClusterName: ecsCluster.clusterName,
                  ServiceName: ecsService.serviceName,
                },
                statistic: 'Average',
                label: 'Memory %',
                period: cdk.Duration.minutes(5),
              }),
            ],
            leftYAxis: { min: 0, max: 100 },
          }),
          new cloudwatch.GraphWidget({
            title: 'RDS - CPU & Connections',
            width: 8,
            height: 6,
            left: [
              new cloudwatch.Metric({
                namespace: 'AWS/RDS',
                metricName: 'CPUUtilization',
                dimensionsMap: { DBInstanceIdentifier: rdsInstance.instanceIdentifier },
                statistic: 'Average',
                label: 'CPU %',
                period: cdk.Duration.minutes(5),
              }),
            ],
            right: [
              new cloudwatch.Metric({
                namespace: 'AWS/RDS',
                metricName: 'DatabaseConnections',
                dimensionsMap: { DBInstanceIdentifier: rdsInstance.instanceIdentifier },
                statistic: 'Average',
                label: 'Connections',
                period: cdk.Duration.minutes(5),
              }),
            ],
          }),
          new cloudwatch.GraphWidget({
            title: 'API Gateway - Requests, Errors, Latency',
            width: 8,
            height: 6,
            left: [
              new cloudwatch.Metric({
                namespace: 'AWS/ApiGateway',
                metricName: 'Count',
                dimensionsMap: { ApiId: httpApi.apiId, Stage: '$default' },
                statistic: 'Sum',
                label: 'Requests',
                period: cdk.Duration.minutes(5),
              }),
            ],
            right: [
              new cloudwatch.Metric({
                namespace: 'AWS/ApiGateway',
                metricName: '5xx',
                dimensionsMap: { ApiId: httpApi.apiId, Stage: '$default' },
                statistic: 'Sum',
                label: '5xx',
                color: '#d62728',
                period: cdk.Duration.minutes(5),
              }),
              new cloudwatch.Metric({
                namespace: 'AWS/ApiGateway',
                metricName: 'Latency',
                dimensionsMap: { ApiId: httpApi.apiId, Stage: '$default' },
                statistic: 'Average',
                label: 'Latency (ms)',
                period: cdk.Duration.minutes(5),
              }),
            ],
          }),
        ],
        [
          new cloudwatch.GraphWidget({
            title: 'ALB - Healthy vs Unhealthy Hosts',
            width: 12,
            height: 6,
            left: [
              new cloudwatch.Metric({
                namespace: 'AWS/ApplicationELB',
                metricName: 'HealthyHostCount',
                dimensionsMap: {
                  LoadBalancer: alb.loadBalancerFullName,
                  TargetGroup: albTargetGroup.targetGroupFullName,
                },
                statistic: 'Average',
                label: 'Healthy',
                period: cdk.Duration.minutes(1),
              }),
              new cloudwatch.Metric({
                namespace: 'AWS/ApplicationELB',
                metricName: 'UnHealthyHostCount',
                dimensionsMap: {
                  LoadBalancer: alb.loadBalancerFullName,
                  TargetGroup: albTargetGroup.targetGroupFullName,
                },
                statistic: 'Average',
                label: 'Unhealthy',
                color: '#d62728',
                period: cdk.Duration.minutes(1),
              }),
            ],
          }),
          new cloudwatch.GraphWidget({
            title: 'Ingestion Lambda - Invocations & Errors',
            width: 12,
            height: 6,
            left: [
              new cloudwatch.Metric({
                namespace: 'AWS/Lambda',
                metricName: 'Invocations',
                dimensionsMap: { FunctionName: lambdaFunction.functionName },
                statistic: 'Sum',
                label: 'Invocations',
                period: cdk.Duration.hours(24),
              }),
              new cloudwatch.Metric({
                namespace: 'AWS/Lambda',
                metricName: 'Errors',
                dimensionsMap: { FunctionName: lambdaFunction.functionName },
                statistic: 'Sum',
                label: 'Errors',
                color: '#d62728',
                period: cdk.Duration.hours(24),
              }),
            ],
          }),
        ],
      ],
    });
  }
}
