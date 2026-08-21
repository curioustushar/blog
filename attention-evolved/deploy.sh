#!/bin/bash
set -e

echo "Building attention-evolved for production..."
npm run build

echo "Copying built files to Hugo static folder..."
rm -rf ../static/attention-evolved
cp -r out ../static/attention-evolved

echo "Done! The app will be available at /blog/attention-evolved/ when the Hugo site is deployed."
echo "Run 'hugo' from the root to build the full site."
