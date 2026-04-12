import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as apigwv2 from 'aws-cdk-lib/aws-apigatewayv2';
import * as apigwv2_integrations from 'aws-cdk-lib/aws-apigatewayv2-integrations';
import * as elbv2 from 'aws-cdk-lib/aws-elasticloadbalancingv2';
import * as logs from 'aws-cdk-lib/aws-logs';
import { Construct } from 'constructs';

export interface ApiConstructProps {
  prefix: string;
  vpc: ec2.Vpc;
  apigwSecurityGroup: ec2.SecurityGroup;
  isolatedSubnets: ec2.ISubnet[];
  albListener: elbv2.ApplicationListener;
  throttleBurstLimit: number;
  throttleRateLimit: number;
}

/**
 * HTTP API Gateway (v2) with a VPC Link pointing at the internal ALB.
 * All traffic is proxied via a single $default route — no auth at the
 * gateway layer; the FastAPI middleware handles admin key validation.
 */
export class ApiConstruct extends Construct {
  public readonly httpApi: apigwv2.HttpApi;
  public readonly httpApiStage: apigwv2.HttpStage;

  constructor(scope: Construct, id: string, props: ApiConstructProps) {
    super(scope, id);
    const {
      prefix,
      vpc,
      apigwSecurityGroup,
      isolatedSubnets,
      albListener,
      throttleBurstLimit,
      throttleRateLimit,
    } = props;

    const accessLogGroup = new logs.LogGroup(this, 'ApiGwLogGroup', {
      logGroupName: `/aws/apigateway/${prefix}-api`,
      retention: logs.RetentionDays.ONE_MONTH,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });
    cdk.Tags.of(accessLogGroup).add('Name', `${prefix}-api-gateway-logs`);

    const vpcLink = new apigwv2.VpcLink(this, 'VpcLink', {
      vpcLinkName: `${prefix}-api`,
      vpc,
      subnets: { subnets: isolatedSubnets },
      securityGroups: [apigwSecurityGroup],
    });
    cdk.Tags.of(vpcLink).add('Name', `${prefix}-api-vpc-link`);

    this.httpApi = new apigwv2.HttpApi(this, 'HttpApi', {
      apiName: `${prefix}-api`,
      createDefaultStage: false,
    });
    cdk.Tags.of(this.httpApi).add('Name', `${prefix}-api-gateway`);

    const integration = new apigwv2_integrations.HttpAlbIntegration(
      'AlbIntegration',
      albListener,
      {
        vpcLink,
        method: apigwv2.HttpMethod.ANY,
        parameterMapping: new apigwv2.ParameterMapping(),
      },
    );

    this.httpApi.addRoutes({
      path: '/{proxy+}',
      methods: [apigwv2.HttpMethod.ANY],
      integration,
    });

    // Catch-all default route
    this.httpApi.addRoutes({
      path: '/',
      methods: [apigwv2.HttpMethod.ANY],
      integration,
    });

    this.httpApiStage = new apigwv2.HttpStage(this, 'DefaultStage', {
      httpApi: this.httpApi,
      stageName: '$default',
      autoDeploy: true,
      throttle: {
        burstLimit: throttleBurstLimit,
        rateLimit: throttleRateLimit,
      },
    });

    // Access logging via CfnStage override (CDK HttpStage doesn't expose this yet)
    const cfnStage = this.httpApiStage.node.defaultChild as apigwv2.CfnStage;
    cfnStage.accessLogSettings = {
      destinationArn: accessLogGroup.logGroupArn,
      format: JSON.stringify({
        requestId: '$context.requestId',
        sourceIp: '$context.identity.sourceIp',
        requestTime: '$context.requestTime',
        httpMethod: '$context.httpMethod',
        routeKey: '$context.routeKey',
        status: '$context.status',
        protocol: '$context.protocol',
        responseLength: '$context.responseLength',
        integrationErr: '$context.integrationErrorMessage',
      }),
    };
    cfnStage.defaultRouteSettings = {
      detailedMetricsEnabled: true,
      throttlingBurstLimit: throttleBurstLimit,
      throttlingRateLimit: throttleRateLimit,
    };
    cdk.Tags.of(this.httpApiStage).add('Name', `${prefix}-api-stage`);
  }
}
