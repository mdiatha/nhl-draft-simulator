#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';
import { NhlDraftStack } from '../lib/nhl-draft-stack';

const app = new cdk.App();

const env = {
  account: process.env.CDK_DEFAULT_ACCOUNT,
  region: process.env.CDK_DEFAULT_REGION ?? 'us-east-1',
};

new NhlDraftStack(app, 'NhlDraftStack', {
  env,
  description: 'NHL Draft Simulator — full-stack AWS infrastructure',
  tags: {
    Project: 'nhl-draft-simulator',
    Environment: app.node.tryGetContext('environment') ?? 'production',
    ManagedBy: 'cdk',
  },
});
