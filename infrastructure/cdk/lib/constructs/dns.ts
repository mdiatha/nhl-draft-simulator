import * as cdk from 'aws-cdk-lib';
import * as route53 from 'aws-cdk-lib/aws-route53';
import * as route53_targets from 'aws-cdk-lib/aws-route53-targets';
import * as acm from 'aws-cdk-lib/aws-certificatemanager';
import * as cloudfront from 'aws-cdk-lib/aws-cloudfront';
import * as apigwv2 from 'aws-cdk-lib/aws-apigatewayv2';
import { Construct } from 'constructs';

export interface DnsConstructProps {
  prefix: string;
  route53ZoneName: string;
  frontendDomainName?: string;
  apiDomainName?: string;
  cloudfrontDistribution: cloudfront.Distribution;
  httpApi: apigwv2.HttpApi;
  httpApiStage: apigwv2.HttpStage;
}

/**
 * Optional Route 53 + ACM custom domains.
 * Only instantiated when route53ZoneName is set.
 * Frontend → CloudFront alias record.
 * API → API Gateway custom domain + regional ACM cert.
 */
export class DnsConstruct extends Construct {
  constructor(scope: Construct, id: string, props: DnsConstructProps) {
    super(scope, id);
    const {
      prefix,
      route53ZoneName,
      frontendDomainName,
      apiDomainName,
      cloudfrontDistribution,
      httpApi,
      httpApiStage,
    } = props;

    const hostedZone = route53.HostedZone.fromLookup(this, 'HostedZone', {
      domainName: route53ZoneName,
    });

    // ── Frontend custom domain ─────────────────────────────────────────────────
    if (frontendDomainName) {
      new route53.ARecord(this, 'FrontendAliasRecord', {
        zone: hostedZone,
        recordName: frontendDomainName,
        target: route53.RecordTarget.fromAlias(
          new route53_targets.CloudFrontTarget(cloudfrontDistribution),
        ),
      });
    }

    // ── API custom domain ──────────────────────────────────────────────────────
    if (apiDomainName) {
      const apiCert = new acm.Certificate(this, 'ApiCert', {
        domainName: apiDomainName,
        validation: acm.CertificateValidation.fromDns(hostedZone),
      });
      cdk.Tags.of(apiCert).add('Name', `${prefix}-api-cert`);

      const apiDomain = new apigwv2.DomainName(this, 'ApiDomainName', {
        domainName: apiDomainName,
        certificate: apiCert,
      });
      cdk.Tags.of(apiDomain).add('Name', `${prefix}-api-domain`);

      new apigwv2.ApiMapping(this, 'ApiMapping', {
        api: httpApi,
        domainName: apiDomain,
        stage: httpApiStage,
      });

      new route53.ARecord(this, 'ApiAliasRecord', {
        zone: hostedZone,
        recordName: apiDomainName,
        target: route53.RecordTarget.fromAlias(
          new route53_targets.ApiGatewayv2DomainProperties(
            apiDomain.regionalDomainName,
            apiDomain.regionalHostedZoneId,
          ),
        ),
      });
    }
  }
}
