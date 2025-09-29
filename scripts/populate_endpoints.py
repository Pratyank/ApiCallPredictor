#!/usr/bin/env python3
"""
Populate endpoint database with sample API specifications
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.spec_parser import OpenAPISpecParser
from tests.fixtures.sample_data import get_stripe_api_spec, get_github_api_spec

async def populate_from_sample_specs():
    """Populate database with sample OpenAPI specifications"""
    
    # Sample OpenAPI specs
    sample_specs = {
        "stripe-billing": get_stripe_api_spec(),
        "github-api": get_github_api_spec()
    }
    
    async with OpenAPISpecParser() as parser:
        for spec_name, spec_data in sample_specs.items():
            print(f"Processing {spec_name}...")
            
            # Extract endpoints from specification
            endpoints = await parser.extract_api_endpoints(spec_data)
            print(f"  Extracted {len(endpoints)} endpoints")
            
            # Store endpoints in database
            fake_url = f"sample://{spec_name}"
            fake_content = f"Sample {spec_name} specification"
            await parser._store_parsed_endpoints(endpoints, fake_url, fake_content)
            
            print(f"  Stored endpoints for {spec_name}")
    
    print(f"\n✅ Successfully populated database with sample API endpoints")

async def populate_billing_endpoints():
    """Add specific billing/invoice endpoints"""
    
    billing_endpoints = [
        {
            "method": "GET",
            "path": "/invoices",
            "summary": "List all invoices",
            "description": "Retrieve a list of all invoices for the account",
            "operation_id": "listInvoices",
            "tags": ["billing", "invoices"]
        },
        {
            "method": "POST", 
            "path": "/invoices",
            "summary": "Create an invoice",
            "description": "Create a new invoice for a customer",
            "operation_id": "createInvoice",
            "tags": ["billing", "invoices"]
        },
        {
            "method": "GET",
            "path": "/invoices/{invoice_id}",
            "summary": "Get invoice details",
            "description": "Retrieve details for a specific invoice",
            "operation_id": "getInvoice", 
            "tags": ["billing", "invoices"]
        },
        {
            "method": "PUT",
            "path": "/invoices/{invoice_id}/status",
            "summary": "Update invoice status",
            "description": "Update the status of an invoice (draft, open, paid, etc.)",
            "operation_id": "updateInvoiceStatus",
            "tags": ["billing", "invoices"]
        },
        {
            "method": "POST",
            "path": "/invoices/{invoice_id}/finalize",
            "summary": "Finalize invoice",
            "description": "Finalize a draft invoice for billing",
            "operation_id": "finalizeInvoice",
            "tags": ["billing", "invoices"]
        },
        {
            "method": "POST",
            "path": "/billing/calculate",
            "summary": "Calculate billing",
            "description": "Calculate billing amounts for a period",
            "operation_id": "calculateBilling",
            "tags": ["billing"]
        },
        {
            "method": "GET",
            "path": "/billing/summary",
            "summary": "Get billing summary", 
            "description": "Get billing summary for a period",
            "operation_id": "getBillingSummary",
            "tags": ["billing"]
        },
        {
            "method": "POST",
            "path": "/billing/process",
            "summary": "Process billing",
            "description": "Process billing for the current period",
            "operation_id": "processBilling",
            "tags": ["billing"]
        }
    ]
    
    async with OpenAPISpecParser() as parser:
        print("Adding billing-specific endpoints...")
        
        # Store billing endpoints
        fake_url = "sample://billing-endpoints"
        fake_content = "Billing-specific API endpoints"
        await parser._store_parsed_endpoints(billing_endpoints, fake_url, fake_content)
        
        print(f"✅ Added {len(billing_endpoints)} billing endpoints")

async def main():
    """Main function to populate endpoints"""
    print("🚀 Populating endpoint database...")
    print()
    
    # Method 1: Load sample OpenAPI specs
    await populate_from_sample_specs()
    print()
    
    # Method 2: Add specific billing endpoints  
    await populate_billing_endpoints()
    print()
    
    # Show final counts
    import sqlite3
    conn = sqlite3.connect('data/cache.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM parsed_endpoints')
    total_count = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(DISTINCT spec_url) FROM parsed_endpoints')
    spec_count = cursor.fetchone()[0]
    
    print(f"📊 Database now contains:")
    print(f"   • {total_count} total endpoints")
    print(f"   • {spec_count} API specifications")
    
    # Show sample of billing endpoints
    cursor.execute('''
        SELECT method, path, summary 
        FROM parsed_endpoints 
        WHERE tags LIKE '%billing%' 
        LIMIT 5
    ''')
    
    billing_samples = cursor.fetchall()
    if billing_samples:
        print(f"   • Sample billing endpoints:")
        for method, path, summary in billing_samples:
            print(f"     - {method} {path} - {summary}")
    
    conn.close()
    print()
    print("✅ Endpoint population completed!")

if __name__ == "__main__":
    asyncio.run(main())