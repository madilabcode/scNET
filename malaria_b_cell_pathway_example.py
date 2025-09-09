#!/usr/bin/env python3
"""
scNET Malaria B Cell Pathway Reconstruction Example

This script provides a basic example of how to use scNET for pathway reconstruction
analysis on malaria-associated B cell datasets.

Usage:
    python malaria_b_cell_pathway_example.py --data_path <path_to_data.h5ad>

Requirements:
    - scNET installed (pip install scnet)
    - Malaria-associated B cell dataset in AnnData format
"""

import argparse
import sys
import os
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import scNET
try:
    import scNET
    print("✓ scNET imported successfully")
except ImportError:
    print("❌ scNET not found. Install with: pip install scnet")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Run scNET pathway reconstruction analysis on malaria B cell data"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to malaria B cell dataset (.h5ad format)"
    )
    parser.add_argument(
        "--project_name", 
        type=str,
        default="malaria_b_cell_analysis",
        help="Name for the analysis project"
    )
    parser.add_argument(
        "--max_epoch",
        type=int,
        default=250,
        help="Maximum training epochs for scNET"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🧬 scNET Malaria B Cell Pathway Reconstruction Analysis")
    print("=" * 60)
    
    # Step 1: Load dataset
    print("📂 Loading dataset...")
    try:
        adata = sc.read_h5ad(args.data_path)
        print(f"✓ Dataset loaded: {adata.shape[0]} cells, {adata.shape[1]} genes")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("Please ensure your dataset is in AnnData (.h5ad) format")
        sys.exit(1)
    
    # Basic dataset info
    print(f"📊 Dataset information:")
    print(f"   - Cells: {adata.n_obs:,}")
    print(f"   - Genes: {adata.n_vars:,}")
    if 'cell_type' in adata.obs.columns:
        print(f"   - Cell types: {adata.obs['cell_type'].nunique()}")
    
    # Step 2: Basic preprocessing check
    print("\n🔧 Checking data preparation...")
    if adata.X.min() < 0:
        print("⚠️  Warning: Negative values detected. Consider log-normalizing data.")
    
    # Step 3: Run scNET analysis
    print(f"\n🚀 Running scNET analysis (project: {args.project_name})...")
    try:
        scNET.run_scNET(
            adata,
            pre_processing_flag=True,
            human_flag=True,
            number_of_batches=3,
            split_cells=True,
            max_epoch=args.max_epoch,
            model_name=args.project_name
        )
        print("✓ scNET training completed successfully")
    except Exception as e:
        print(f"❌ Error during scNET training: {e}")
        sys.exit(1)
    
    # Step 4: Load embeddings and create reconstructed object
    print("\n📈 Loading scNET results...")
    try:
        embedded_genes, embedded_cells, node_features, out_features = scNET.load_embeddings(args.project_name)
        recon_obj = scNET.create_reconstructed_obj(node_features, out_features, adata)
        print("✓ Reconstructed object created successfully")
        print(f"   - Shape: {recon_obj.shape}")
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        sys.exit(1)
    
    # Step 5: Define B cell subtypes (example mapping)
    print("\n🏷️  Annotating B cell subtypes...")
    b_cell_types = {
        "0": "Memory B cells",
        "1": "Naive B cells",
        "2": "Atypical Memory B cells",
        "3": "Plasma cells",
        "4": "Germinal Center B cells",
        "5": "Activated B cells",
        "6": "Transitional B cells",
        "7": "Regulatory B cells"
    }
    
    # Map cluster IDs to cell types
    recon_obj.obs["B_Cell_Subtype"] = recon_obj.obs.leiden.map(b_cell_types)
    recon_obj.obs["B_Cell_Subtype"] = recon_obj.obs["B_Cell_Subtype"].fillna("Unknown")
    
    print(f"✓ Cell type distribution:")
    for cell_type, count in recon_obj.obs["B_Cell_Subtype"].value_counts().items():
        print(f"   - {cell_type}: {count} cells")
    
    # Step 6: Pathway reconstruction analysis
    print("\n🧪 Running pathway reconstruction analysis...")
    target_subtypes = ["Memory B cells", "Atypical Memory B cells", "Plasma cells", "Activated B cells"]
    available_subtypes = [st for st in target_subtypes if st in recon_obj.obs["B_Cell_Subtype"].values]
    
    if len(available_subtypes) == 0:
        print("⚠️  Warning: No target B cell subtypes found. Using all available subtypes.")
        available_subtypes = recon_obj.obs["B_Cell_Subtype"].unique()[:4]  # Use first 4 subtypes
    
    print(f"📊 Analyzing subtypes: {available_subtypes}")
    
    try:
        subset_data = recon_obj[recon_obj.obs["B_Cell_Subtype"].isin(available_subtypes)].copy()
        
        de_genes_per_group, significant_pathways, filtered_kegg, enrichment_results = scNET.pathway_enricment(
            subset_data,
            groupby="B_Cell_Subtype",
            logfc_threshold=0.25,
            pval_threshold=0.05
        )
        print("✓ Pathway enrichment analysis completed")
        
        # Count results
        total_pathways = sum(len(df) for df in significant_pathways.values())
        print(f"   - Total significant pathways: {total_pathways}")
        
        for subtype, pathways_df in significant_pathways.items():
            print(f"   - {subtype}: {len(pathways_df)} pathways")
            
    except Exception as e:
        print(f"❌ Error in pathway analysis: {e}")
        print("This might be due to insufficient cell numbers or data quality issues")
        sys.exit(1)
    
    # Step 7: Visualize results
    print("\n📊 Generating pathway visualization...")
    try:
        plt.figure(figsize=(12, 8))
        scNET.plot_de_pathways(significant_pathways, enrichment_results, head=20)
        plt.title('Malaria B Cell Pathway Reconstruction: Top Enriched Pathways', fontsize=14)
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(args.output_dir, f"{args.project_name}_pathways.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Pathway plot saved to: {plot_path}")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not generate pathway plot: {e}")
    
    # Step 8: Save key results
    print("\n💾 Saving results...")
    
    # Save significant pathways summary
    summary_data = []
    for subtype, pathways_df in significant_pathways.items():
        for idx, row in pathways_df.head(10).iterrows():
            summary_data.append({
                'B_Cell_Subtype': subtype,
                'Pathway': row['Term'],
                'Adjusted_P_Value': row['Adjusted P-value']
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(args.output_dir, f"{args.project_name}_pathway_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"✓ Pathway summary saved to: {summary_path}")
    
    # Step 9: Display top malaria-relevant pathways
    print("\n🦠 Malaria-relevant pathways identified:")
    malaria_keywords = ['B cell receptor', 'antigen', 'complement', 'cytokine', 'NF-kappa', 'JAK-STAT', 'toll-like']
    
    for subtype, enrichment_df in enrichment_results.items():
        malaria_pathways = enrichment_df[
            enrichment_df['Term'].str.contains('|'.join(malaria_keywords), case=False, na=False) & 
            (enrichment_df['Adjusted P-value'] < 0.05)
        ].head(3)
        
        if len(malaria_pathways) > 0:
            print(f"\n   {subtype}:")
            for idx, row in malaria_pathways.iterrows():
                print(f"   • {row['Term']} (p={row['Adjusted P-value']:.2e})")
    
    print("\n🎉 Analysis completed successfully!")
    print(f"📁 Results saved in: {args.output_dir}")
    print("\n💡 Next steps:")
    print("   1. Review the pathway summary CSV file")
    print("   2. Examine the pathway enrichment plot")
    print("   3. Validate key findings with literature")
    print("   4. Consider additional targeted analyses")
    print("\n📖 For detailed tutorial, see: scNET_Malaria_B_Cell_Pathway_Reconstruction.ipynb")

if __name__ == "__main__":
    main()