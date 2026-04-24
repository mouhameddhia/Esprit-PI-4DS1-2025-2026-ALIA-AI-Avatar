"""Add test pharmaceutical products to MongoDB."""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add backend to path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from dotenv import load_dotenv
import os

# Load environment variables
env_path = backend_dir / ".env"
load_dotenv(dotenv_path=env_path)

from motor.motor_asyncio import AsyncIOMotorClient


def _utc_now():
    return datetime.now(timezone.utc)


# Test products data
TEST_PRODUCTS = [
    {
        "name": "Cardiostat",
        "description": "A potent ACE inhibitor used for managing hypertension and heart failure. Cardiostat reduces blood pressure by relaxing blood vessels and improving cardiac function.",
        "category": "Cardiovascular",
        "indications": [
            "Essential hypertension",
            "Heart failure (NYHA Class II-IV)",
            "Post-myocardial infarction",
            "Left ventricular dysfunction"
        ],
        "contraindications": [
            "Pregnancy",
            "Bilateral renal artery stenosis",
            "History of angioedema",
            "Severe renal impairment (eGFR < 30)"
        ],
        "dosage": "Initial: 10mg once daily. Maintenance: 20-40mg daily in divided doses. Max: 80mg daily.",
        "created_at": _utc_now(),
        "updated_at": _utc_now()
    },
    {
        "name": "Glycodia",
        "description": "An advanced diabetes medication combining metformin and a DPP-4 inhibitor. Glycodia improves glycemic control through multiple mechanisms of action.",
        "category": "Endocrinology",
        "indications": [
            "Type 2 diabetes mellitus",
            "Inadequate glycemic control on monotherapy",
            "Improved fasting and postprandial glucose levels"
        ],
        "contraindications": [
            "Type 1 diabetes",
            "Severe renal disease",
            "Diabetic ketoacidosis",
            "Heart failure (NYHA Class III-IV)"
        ],
        "dosage": "500/2.5mg twice daily with meals. Can increase to 1000/5mg twice daily based on response.",
        "created_at": _utc_now(),
        "updated_at": _utc_now()
    },
    {
        "name": "Respirex",
        "description": "A long-acting beta-2 agonist with inhaled corticosteroid for COPD and asthma management. Respirex provides sustained bronchodilation and anti-inflammatory benefits.",
        "category": "Respiratory",
        "indications": [
            "Chronic Obstructive Pulmonary Disease (COPD)",
            "Moderate to severe persistent asthma",
            "Prevention of asthma exacerbations",
            "Maintenance bronchodilator therapy"
        ],
        "contraindications": [
            "Acute asthma attacks",
            "Hypersensitivity to beta-agonists",
            "Cardiac arrhythmias",
            "Thyrotoxicosis"
        ],
        "dosage": "Inhale 2 puffs twice daily. Each puff contains 100mcg salbutamol and 50mcg fluticasone.",
        "created_at": _utc_now(),
        "updated_at": _utc_now()
    },
    {
        "name": "Immunoguard",
        "description": "An immunomodulatory agent that enhances immune response for cancer patients. Immunoguard works by activating T-cells and promoting anti-tumor immunity.",
        "category": "Oncology",
        "indications": [
            "Metastatic melanoma",
            "Non-small cell lung cancer (NSCLC)",
            "Adjuvant therapy for high-risk cancers",
            "Immunotherapy-resistant tumors"
        ],
        "contraindications": [
            "Active infection",
            "Severe autoimmune disease",
            "Organ transplant recipients",
            "Pregnancy and lactation"
        ],
        "dosage": "IV infusion: 200mg every 2 weeks for 8 weeks, then every 4 weeks. Adjust based on tolerability.",
        "created_at": _utc_now(),
        "updated_at": _utc_now()
    },
    {
        "name": "Neurocept",
        "description": "A neuroprotective agent for neurodegenerative diseases. Neurocept crosses the blood-brain barrier and reduces neuronal apoptosis.",
        "category": "Neurology",
        "indications": [
            "Parkinson's disease",
            "Mild to moderate Alzheimer's disease",
            "Amyotrophic lateral sclerosis (ALS)",
            "Neuropathic pain management"
        ],
        "contraindications": [
            "Severe hepatic impairment",
            "Uncontrolled epilepsy",
            "Severe renal failure",
            "Concurrent monoamine oxidase inhibitors"
        ],
        "dosage": "Oral: 100mg twice daily with food. Maximum daily dose: 400mg. Adjust in elderly: 50-100mg twice daily.",
        "created_at": _utc_now(),
        "updated_at": _utc_now()
    },
    {
        "name": "Arthcure",
        "description": "A targeted anti-inflammatory for rheumatoid arthritis. Arthcure inhibits TNF-alpha and IL-6, reducing joint inflammation.",
        "category": "Rheumatology",
        "indications": [
            "Active rheumatoid arthritis",
            "Moderate to severe disease",
            "Failed conventional DMARD therapy",
            "Early intervention in RA"
        ],
        "contraindications": [
            "Active tuberculosis",
            "Hepatitis B infection",
            "Neutropenia (ANC < 1500)",
            "Congestive heart failure"
        ],
        "dosage": "SC injection: 50mg once weekly or 25mg twice weekly. Can be combined with methotrexate.",
        "created_at": _utc_now(),
        "updated_at": _utc_now()
    }
]


async def add_test_products():
    """Add test products to MongoDB."""
    mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017/alia")
    
    try:
        client = AsyncIOMotorClient(mongodb_url)
        db = client.alia
        
        print("🔌 Connecting to MongoDB...")
        await client.admin.command('ping')
        print("✓ Connected to MongoDB")
        
        print(f"\n📦 Adding {len(TEST_PRODUCTS)} test products...")
        inserted_count = 0
        updated_count = 0
        product_ids = []

        for product in TEST_PRODUCTS:
            product_doc = {**product, "updated_at": _utc_now()}
            existing = await db.products.find_one({"name": product_doc["name"]}, {"_id": 1, "created_at": 1})

            if existing:
                await db.products.update_one(
                    {"_id": existing["_id"]},
                    {
                        "$set": {
                            **product_doc,
                            "created_at": existing.get("created_at", product_doc["updated_at"]),
                        }
                    },
                )
                updated_count += 1
                product_ids.append(existing["_id"])
            else:
                product_doc["created_at"] = product_doc["updated_at"]
                result = await db.products.insert_one(product_doc)
                inserted_count += 1
                product_ids.append(result.inserted_id)

        print(f"✓ Inserted {inserted_count} new products")
        print(f"✓ Updated {updated_count} existing products")

        print("\nProduct IDs:")
        for i, product_id in enumerate(product_ids, 1):
            print(f"  {i}. {product_id}")
        
        # Verify insertion
        count = await db.products.count_documents({})
        print(f"\n✓ Total products in database: {count}")
        
        # Show product names
        print("\n📋 Products added:")
        products = await db.products.find({}, {"name": 1}).to_list(None)
        for product in products:
            print(f"  • {product['name']}")
        
        client.close()
        print("\n✅ Test products added successfully!")
        print("\n📝 Next steps:")
        print("   1. Start/restart the backend: uvicorn backend.main:app --reload")
        print("   2. It will automatically index products into Pinecone on startup")
        print("   3. Or manually trigger indexing: POST http://localhost:8000/admin/reindex-products")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(add_test_products())
    sys.exit(0 if success else 1)
