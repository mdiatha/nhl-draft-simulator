import * as cdk from 'aws-cdk-lib';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import * as ssm from 'aws-cdk-lib/aws-ssm';
import { Construct } from 'constructs';

export interface SecretsConstructProps {
  prefix: string;
}

export interface SsmParams {
  anthropicApiKey: ssm.StringParameter;
  secretKey: ssm.StringParameter;
  adminApiKey: ssm.StringParameter;
}

/**
 * Secrets Manager holds the DATABASE_URL (generated at deploy time by the Compute construct
 * and written as a post-deploy step or via CDK custom resource).
 *
 * SSM SecureString parameters hold static API keys - free and KMS-encrypted,
 * unlike Secrets Manager which costs $0.40/secret/month.
 *
 * Actual secret values must be set externally (e.g. `aws ssm put-parameter`
 * or via CDK context / environment variables in a pipeline).
 */
export class SecretsConstruct extends Construct {
  public readonly appSecret: secretsmanager.Secret;
  public readonly ssmParams: SsmParams;

  constructor(scope: Construct, id: string, props: SecretsConstructProps) {
    super(scope, id);
    const { prefix } = props;

    // Secrets Manager - DATABASE_URL (populated after RDS is created)
    this.appSecret = new secretsmanager.Secret(this, 'AppSecret', {
      secretName: 'nhl-draft/production',
      description: 'NHL Draft Simulator - DATABASE_URL (loaded by FastAPI at startup)',
      // Initial placeholder; real value set once RDS endpoint is known
      generateSecretString: {
        secretStringTemplate: JSON.stringify({ DATABASE_URL: 'postgresql+psycopg2://placeholder' }),
        generateStringKey: '_unused',
      },
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });
    cdk.Tags.of(this.appSecret).add('Name', `${prefix}-app-secret`);

    // SSM SecureString parameters - static API keys
    // Values must be provided via CDK context or set manually after first deploy.
    const makeParam = (
      logicalId: string,
      name: string,
      description: string,
      contextKey: string,
    ): ssm.StringParameter => {
      const value: string = this.node.tryGetContext(contextKey) ?? 'placeholder';
      return new ssm.StringParameter(this, logicalId, {
        parameterName: name,
        description,
        stringValue: value,
        tier: ssm.ParameterTier.STANDARD,
      });
    };

    this.ssmParams = {
      anthropicApiKey: makeParam(
        'AnthropicApiKey',
        '/nhl-draft/anthropic_api_key',
        'Anthropic API key for Claude scout agent',
        'anthropicApiKey',
      ),
      secretKey: makeParam(
        'SecretKey',
        '/nhl-draft/secret_key',
        'FastAPI application secret key',
        'secretKey',
      ),
      adminApiKey: makeParam(
        'AdminApiKey',
        '/nhl-draft/admin_api_key',
        'Admin API key for protected endpoints (X-Admin-Key header)',
        'adminApiKey',
      ),
    };
  }
}
