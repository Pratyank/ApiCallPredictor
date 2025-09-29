#!/usr/bin/env python3
"""
Quick database population with billing endpoints
"""
import sqlite3
import json
import hashlib
from datetime import datetime

def quick_populate_billing():
    """Directly insert billing endpoints into parsed_endpoints table"""
    
    # Billing endpoints for Q2 finishing scenario
    endpoints = [
        ("GET", "/invoices", "listInvoices", "List all invoices", "Retrieve a list of all invoices", '["billing", "invoices"]'),
        ("POST", "/invoices", "createInvoice", "Create an invoice", "Create a new invoice for a customer", '["billing", "invoices"]'),
        ("GET", "/invoices/{invoice_id}", "getInvoice", "Get invoice details", "Retrieve details for a specific invoice", '["billing", "invoices"]'),
        ("PUT", "/invoices/{invoice_id}/status", "updateInvoiceStatus", "Update invoice status", "Update the status of an invoice", '["billing", "invoices"]'),
        ("POST", "/invoices/{invoice_id}/finalize", "finalizeInvoice", "Finalize invoice", "Finalize a draft invoice for billing", '["billing", "invoices"]'),
        ("POST", "/billing/calculate", "calculateBilling", "Calculate billing", "Calculate billing amounts for a period", '["billing"]'),
        ("GET", "/billing/summary", "getBillingSummary", "Get billing summary", "Get billing summary for a period", '["billing"]'), 
        ("POST", "/billing/process", "processBilling", "Process billing", "Process billing for the current period", '["billing"]'),
        ("GET", "/billing/reports", "getBillingReports", "Get billing reports", "Generate billing reports for analysis", '["billing", "reports"]'),
        ("POST", "/billing/q2/finalize", "finalizeQ2Billing", "Finalize Q2 billing", "Complete and finalize Q2 billing period", '["billing", "quarterly"]'),
    ]
    
    conn = sqlite3.connect('data/cache.db')
    cursor = conn.cursor()
    
    # Check current count
    cursor.execute('SELECT COUNT(*) FROM parsed_endpoints')
    initial_count = cursor.fetchone()[0]
    print(f"Initial parsed_endpoints count: {initial_count}")
    
    # Insert billing endpoints
    spec_url = "quick://billing-endpoints"
    spec_hash = hashlib.md5(spec_url.encode()).hexdigest()
    
    for method, path, operation_id, summary, description, tags in endpoints:
        endpoint_id = f"{method}_{path}".replace("/", "_").replace("{", "").replace("}", "")
        
        cursor.execute('''
            INSERT OR REPLACE INTO parsed_endpoints 
            (endpoint_id, method, path, summary, description, operation_id, tags, spec_url, spec_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (endpoint_id, method, path, summary, description, operation_id, tags, spec_url, spec_hash))
    
    conn.commit()
    
    # Check final count
    cursor.execute('SELECT COUNT(*) FROM parsed_endpoints')
    final_count = cursor.fetchone()[0]
    
    print(f"✅ Added {final_count - initial_count} billing endpoints")
    print(f"Final parsed_endpoints count: {final_count}")
    
    # Show what was added
    cursor.execute('SELECT method, path, summary FROM parsed_endpoints WHERE spec_url = ? LIMIT 5', (spec_url,))
    samples = cursor.fetchall()
    print("\nSample billing endpoints added:")
    for method, path, summary in samples:
        print(f"  {method} {path} - {summary}")
    
    conn.close()

if __name__ == "__main__":
    quick_populate_billing()