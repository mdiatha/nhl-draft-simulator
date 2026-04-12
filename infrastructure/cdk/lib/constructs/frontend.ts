import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as cloudfront from 'aws-cdk-lib/aws-cloudfront';
import * as cloudfront_origins from 'aws-cdk-lib/aws-cloudfront-origins';
import * as acm from 'aws-cdk-lib/aws-certificatemanager';
import * as route53 from 'aws-cdk-lib/aws-route53';
import { Construct } from 'constructs';

export interface FrontendConstructProps {
  prefix: string;
  frontendBucket: s3.Bucket;
  frontendDomainName?: string;
  route53ZoneName?: string;
}

/**
 * CloudFront distribution for the React SPA.
 * Uses Origin Access Control (OAC) — the modern replacement for OAI.
 * 404s are remapped to index.html so React Router handles client-side routes.
 */
export class FrontendConstruct extends Construct {
  public readonly distribution: cloudfront.Distribution;

  constructor(scope: Construct, id: string, props: FrontendConstructProps) {
    super(scope, id);
    const { prefix, frontendBucket, frontendDomainName, route53ZoneName } = props;

    let certificate: acm.DnsValidatedCertificate | undefined;
    const aliases: string[] = [];

    if (frontendDomainName && route53ZoneName) {
      const hostedZone = route53.HostedZone.fromLookup(this, 'HostedZone', {
        domainName: route53ZoneName,
      });

      // CloudFront certificates must be in us-east-1
      certificate = new acm.DnsValidatedCertificate(this, 'FrontendCert', {
        domainName: frontendDomainName,
        hostedZone,
        region: 'us-east-1',
      });
      cdk.Tags.of(certificate).add('Name', `${prefix}-frontend-cert`);
      aliases.push(frontendDomainName);
    }

    this.distribution = new cloudfront.Distribution(this, 'Distribution', {
      comment: 'NHL Draft Simulator — React SPA',
      defaultRootObject: 'index.html',
      domainNames: aliases.length > 0 ? aliases : undefined,
      ...(certificate ? { certificate } : {}),
      defaultBehavior: {
        origin: cloudfront_origins.S3BucketOrigin.withOriginAccessControl(frontendBucket),
        allowedMethods: cloudfront.AllowedMethods.ALLOW_GET_HEAD,
        cachedMethods: cloudfront.CachedMethods.CACHE_GET_HEAD,
        viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
        compress: true,
        cachePolicy: cloudfront.CachePolicy.CACHING_OPTIMIZED,
      },
      errorResponses: [
        // SPA routing: 404 → index.html so React Router handles client-side routes
        {
          httpStatus: 404,
          responseHttpStatus: 200,
          responsePagePath: '/index.html',
          ttl: cdk.Duration.seconds(0),
        },
      ],
      priceClass: cloudfront.PriceClass.PRICE_CLASS_100,
    });
    cdk.Tags.of(this.distribution).add('Name', `${prefix}-cdn`);
  }
}
