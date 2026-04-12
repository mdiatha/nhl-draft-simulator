import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import { Construct } from 'constructs';

export interface NetworkConstructProps {
  prefix: string;
}

/**
 * VPC with no NAT gateway (saves ~$32/mo).
 * ECS container instances live in public subnets with internet via IGW.
 * Internal services (RDS, private ALB, API Gateway VPC link ENIs) live in isolated subnets.
 */
export class NetworkConstruct extends Construct {
  public readonly vpc: ec2.Vpc;
  public readonly publicSubnets: ec2.ISubnet[];
  public readonly isolatedSubnets: ec2.ISubnet[];
  public readonly albSg: ec2.SecurityGroup;
  public readonly apigwVpcLinkSg: ec2.SecurityGroup;
  public readonly ecsSg: ec2.SecurityGroup;
  public readonly rdsSg: ec2.SecurityGroup;

  constructor(scope: Construct, id: string, props: NetworkConstructProps) {
    super(scope, id);
    const { prefix } = props;

    this.vpc = new ec2.Vpc(this, 'Vpc', {
      vpcName: `${prefix}-vpc`,
      ipAddresses: ec2.IpAddresses.cidr('10.0.0.0/16'),
      maxAzs: 2,
      natGateways: 0, // no NAT — saves $32/mo
      subnetConfiguration: [
        {
          name: 'public',
          subnetType: ec2.SubnetType.PUBLIC,
          cidrMask: 24,
          mapPublicIpOnLaunch: true,
        },
        {
          name: 'isolated',
          subnetType: ec2.SubnetType.PRIVATE_ISOLATED,
          cidrMask: 24,
        },
      ],
    });

    this.publicSubnets = this.vpc.publicSubnets;
    this.isolatedSubnets = this.vpc.isolatedSubnets;

    // ── Security groups ────────────────────────────────────────────────────────

    // API Gateway VPC link ENIs — egress only
    this.apigwVpcLinkSg = new ec2.SecurityGroup(this, 'ApigwVpcLinkSg', {
      vpc: this.vpc,
      securityGroupName: `${prefix}-apigw-vpc-link`,
      description: 'API Gateway VPC link ENIs',
      allowAllOutbound: true,
    });
    cdk.Tags.of(this.apigwVpcLinkSg).add('Name', `${prefix}-apigw-vpc-link-sg`);

    // Private ALB — HTTP inbound from API Gateway VPC link only
    this.albSg = new ec2.SecurityGroup(this, 'AlbSg', {
      vpc: this.vpc,
      securityGroupName: `${prefix}-alb`,
      description: 'Private ALB — HTTP inbound from API Gateway VPC link only',
      allowAllOutbound: true,
    });
    this.albSg.addIngressRule(
      ec2.Peer.securityGroupId(this.apigwVpcLinkSg.securityGroupId),
      ec2.Port.tcp(80),
      'HTTP from API Gateway VPC link',
    );
    cdk.Tags.of(this.albSg).add('Name', `${prefix}-alb-sg`);

    // ECS instances — app port inbound from ALB only
    this.ecsSg = new ec2.SecurityGroup(this, 'EcsSg', {
      vpc: this.vpc,
      securityGroupName: `${prefix}-ecs`,
      description: 'ECS container instances — app port inbound from ALB only',
      allowAllOutbound: true,
    });
    this.ecsSg.addIngressRule(
      ec2.Peer.securityGroupId(this.albSg.securityGroupId),
      ec2.Port.tcp(8000),
      'FastAPI from ALB',
    );
    cdk.Tags.of(this.ecsSg).add('Name', `${prefix}-ecs-sg`);

    // RDS — PostgreSQL inbound from ECS only, no outbound
    this.rdsSg = new ec2.SecurityGroup(this, 'RdsSg', {
      vpc: this.vpc,
      securityGroupName: `${prefix}-rds`,
      description: 'PostgreSQL — inbound from app tier only, no outbound',
      allowAllOutbound: false,
    });
    this.rdsSg.addIngressRule(
      ec2.Peer.securityGroupId(this.ecsSg.securityGroupId),
      ec2.Port.tcp(5432),
      'PostgreSQL from ECS instances',
    );
    cdk.Tags.of(this.rdsSg).add('Name', `${prefix}-rds-sg`);
  }
}
