import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import { Construct } from 'constructs';

export interface StorageConstructProps {
  prefix: string;
}

export class StorageConstruct extends Construct {
  public readonly modelsBucket: s3.Bucket;
  public readonly frontendBucket: s3.Bucket;

  constructor(scope: Construct, id: string, props: StorageConstructProps) {
    super(scope, id);
    const { prefix } = props;

    // ML model artifact storage
    this.modelsBucket = new s3.Bucket(this, 'ModelsBucket', {
      // Account ID appended automatically via physical name token
      bucketName: cdk.PhysicalName.GENERATE_IF_NEEDED,
      versioned: true,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      lifecycleRules: [
        {
          id: 'keep-10-noncurrent-versions',
          enabled: true,
          noncurrentVersionExpiration: cdk.Duration.days(30),
          noncurrentVersionsToRetain: 10,
        },
      ],
    });
    cdk.Tags.of(this.modelsBucket).add('Name', `${prefix}-models`);

    // React frontend static hosting - served via CloudFront OAC
    this.frontendBucket = new s3.Bucket(this, 'FrontendBucket', {
      bucketName: cdk.PhysicalName.GENERATE_IF_NEEDED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });
    cdk.Tags.of(this.frontendBucket).add('Name', `${prefix}-frontend`);
  }
}
