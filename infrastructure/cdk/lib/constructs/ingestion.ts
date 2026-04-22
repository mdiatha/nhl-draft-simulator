import * as cdk from 'aws-cdk-lib';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as sqs from 'aws-cdk-lib/aws-sqs';
import * as events from 'aws-cdk-lib/aws-events';
import * as events_targets from 'aws-cdk-lib/aws-events-targets';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as ssm from 'aws-cdk-lib/aws-ssm';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as path from 'path';
import { Construct } from 'constructs';

export interface IngestionConstructProps {
  prefix: string;
  apiEndpoint: string;
  adminApiKeyParam: ssm.IStringParameter;
}

/**
 * Daily ingestion trigger:
 *   EventBridge cron (06:00 UTC) → Lambda → FastAPI /api/admin/ingest via API Gateway
 * The handler is stdlib-only Python so no Lambda layer is needed.
 */
export class IngestionConstruct extends Construct {
  public readonly lambdaFunction: lambda.Function;

  constructor(scope: Construct, id: string, props: IngestionConstructProps) {
    super(scope, id);
    const { prefix, apiEndpoint, adminApiKeyParam } = props;

    const logGroup = new logs.LogGroup(this, 'LambdaLogGroup', {
      logGroupName: `/aws/lambda/${prefix}-ingestion-trigger`,
      retention: logs.RetentionDays.ONE_MONTH,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });
    cdk.Tags.of(logGroup).add('Name', `${prefix}-lambda-logs`);

    const dlq = new sqs.Queue(this, 'DLQ', {
      queueName: `${prefix}-lambda-dlq`,
      retentionPeriod: cdk.Duration.days(14),
    });
    cdk.Tags.of(dlq).add('Name', `${prefix}-lambda-dlq`);

    this.lambdaFunction = new lambda.Function(this, 'IngestionLambda', {
      functionName: `${prefix}-ingestion-trigger`,
      description:
        'Daily NHL ingestion trigger - EventBridge fires this, calls FastAPI /api/admin/ingest via API Gateway',
      runtime: lambda.Runtime.PYTHON_3_11,
      handler: 'handler.handler',
      // Points to lambda/ingestion/ relative to the repo root
      code: lambda.Code.fromAsset(
        path.join(__dirname, '..', '..', '..', '..', 'lambda', 'ingestion'),
      ),
      memorySize: 128,
      timeout: cdk.Duration.seconds(60),
      environment: {
        API_BASE_URL: apiEndpoint,
        // ADMIN_API_KEY_PARAM holds the SSM parameter name; the handler reads
        // the actual secret value at invocation time via boto3 so it is never
        // baked into the function configuration.
        ADMIN_API_KEY_PARAM: adminApiKeyParam.parameterName,
      },
      deadLetterQueue: dlq,
      logGroup,
    });
    cdk.Tags.of(this.lambdaFunction).add('Name', `${prefix}-ingestion-trigger`);

    // DLQ send permission for Lambda's execution role
    dlq.grantSendMessages(this.lambdaFunction);

    // Allow Lambda to read the admin API key from SSM at invocation time
    adminApiKeyParam.grantRead(this.lambdaFunction);

    // EventBridge daily cron - 06:00 UTC
    const rule = new events.Rule(this, 'DailyIngestionRule', {
      ruleName: `${prefix}-daily-ingestion`,
      description: 'Triggers NHL data ingestion Lambda daily at 06:00 UTC',
      schedule: events.Schedule.cron({ minute: '0', hour: '6' }),
    });
    cdk.Tags.of(rule).add('Name', `${prefix}-daily-ingestion-rule`);

    rule.addTarget(
      new events_targets.LambdaFunction(this.lambdaFunction, {
        event: events.RuleTargetInput.fromObject({ triggered_by: 'cron' }),
      }),
    );
  }
}
