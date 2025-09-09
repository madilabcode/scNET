# Malaria-Associated B Cell Dataset and Pathway Reconstruction

This document provides detailed information about the malaria-associated B cell dataset mentioned in the scNET paper and instructions for pathway reconstruction analysis.

## Dataset Information

### Paper Reference Dataset
The scNET paper mentions a human malaria-associated B cell dataset with the following characteristics:
- **Genes**: 19,374 genes
- **Cells**: 7,044 cells
- **Cell Type**: Human B cells from malaria-exposed individuals
- **Focus**: Pathway reconstruction in malaria immune response

### Dataset Availability Note
The exact dataset from the paper may not be directly available in public repositories. Users have reported finding similar datasets with 5,056 cells instead of the mentioned 7,044 cells. This is common as datasets may be:
- Updated or reprocessed after publication
- Available in different versions
- Subject to quality control filtering

## Alternative Dataset Sources

Since the exact dataset may not be available, here are recommended sources for similar malaria-associated B cell datasets:

### 1. Gene Expression Omnibus (GEO)
- Search terms: "malaria B cell single cell", "Plasmodium immune response scRNA-seq"
- Look for studies on malaria vaccine responses
- Focus on datasets with B cell populations

### 2. Single Cell Portal (Broad Institute)
- Browse malaria-related immune studies
- Look for datasets from malaria-endemic regions
- Check for longitudinal malaria exposure studies

### 3. European Bioinformatics Institute (EBI)
- Single Cell Expression Atlas
- Search for immune system responses to parasites
- Filter for human B cell datasets

### 4. ArrayExpress
- Search for malaria scRNA-seq studies
- Look for immune profiling studies

### 5. Published Studies
Check these types of publications for associated datasets:
- Malaria vaccine trials with immune profiling
- Studies of acquired malaria immunity
- B cell memory formation in malaria
- Atypical memory B cell studies in malaria

## Dataset Requirements

For successful pathway reconstruction analysis, your dataset should have:

### Minimum Requirements
- **Cells**: At least 1,000 B cells (preferably 3,000+)
- **Genes**: 15,000+ genes after filtering
- **Quality**: Good quality scRNA-seq data with proper QC
- **Metadata**: Cell type annotations or clustering information

### Optimal Characteristics
- **Size**: 5,000-10,000 cells, 18,000-20,000 genes
- **Coverage**: Multiple B cell subtypes represented
- **Context**: Malaria exposure or vaccine response context
- **Depth**: Sufficient sequencing depth for pathway analysis

### Data Format
- **Primary**: AnnData (.h5ad) format
- **Alternative**: 10x formats, CSV matrices, Seurat objects
- **Metadata**: Cell type, sample information, treatment conditions

## Pathway Reconstruction Tutorial

The comprehensive pathway reconstruction tutorial is available in:
**scNET_Malaria_B_Cell_Pathway_Reconstruction.ipynb**

This tutorial covers:

### 1. Data Preparation
- Loading malaria-associated B cell datasets
- Quality control and filtering
- Preprocessing for scNET analysis

### 2. scNET Analysis
- Training scNET with appropriate parameters
- Generating enhanced gene and cell embeddings
- Creating reconstructed expression matrices

### 3. B Cell Subtype Identification
- Clustering and annotation of B cell subtypes
- Focus on malaria-relevant populations:
  - Memory B cells
  - Atypical memory B cells
  - Plasma cells
  - Activated B cells
  - Naive B cells

### 4. Pathway Reconstruction
- Differential expression analysis per subtype
- KEGG pathway enrichment analysis
- Focus on malaria-relevant pathways:
  - B cell receptor signaling
  - Antigen processing and presentation
  - Complement cascade
  - Cytokine signaling
  - NF-κB pathway
  - JAK-STAT signaling

### 5. Network Analysis
- Building co-embedded gene networks
- Identifying gene modules
- Finding hub genes and key regulators

### 6. Visualization and Interpretation
- Pathway enrichment heatmaps
- Network visualizations
- Biological interpretation

## Key Pathways in Malaria B Cell Response

The pathway reconstruction analysis focuses on these malaria-relevant pathways:

### Primary Immune Pathways
1. **B cell receptor signaling pathway**
2. **Antigen processing and presentation**
3. **Complement and coagulation cascades**
4. **Cytokine-cytokine receptor interaction**

### Regulatory Pathways
5. **NF-kappa B signaling pathway**
6. **JAK-STAT signaling pathway**
7. **Toll-like receptor signaling pathway**
8. **IL-17 signaling pathway**

### Effector Pathways
9. **TNF signaling pathway**
10. **Fc gamma R-mediated phagocytosis**

## Expected Results

### B Cell Subtype Signatures
- **Memory B cells**: Enhanced BCR signaling, antigen presentation
- **Atypical Memory B cells**: Unique activation signatures, exhaustion markers
- **Plasma cells**: Antibody production, ER stress pathways
- **Activated B cells**: Proliferation, inflammatory responses

### Clinical Relevance
- Understanding malaria vaccine responses
- Identifying protective immune signatures
- Biomarker discovery for malaria resistance
- Insights into immune memory formation

## Usage Instructions

### Quick Start with Example Script

For a rapid analysis of your malaria B cell dataset, use the provided example script:

```bash
# First, set up the scNET environment
conda env create -f ./scNET/Data/scNET-env.yml
conda activate scNET

# Or install scNET via pip
pip install scnet

# Run the pathway reconstruction analysis
python malaria_b_cell_pathway_example.py --data_path your_dataset.h5ad --project_name my_analysis
```

This script provides:
- Automated scNET analysis workflow
- B cell subtype annotation
- Pathway reconstruction analysis
- Results visualization and export

### Detailed Tutorial

1. **Obtain Dataset**: Download a suitable malaria-associated B cell dataset
2. **Prepare Environment**: Install scNET and dependencies
3. **Follow Tutorial**: Use the step-by-step notebook guide
4. **Customize Analysis**: Adapt parameters for your specific dataset
5. **Interpret Results**: Focus on malaria-relevant biological insights

## Support and Questions

If you need help with:
- **Dataset access**: Check the sources listed above or contact the original authors
- **Technical issues**: Create an issue in the scNET GitHub repository
- **Biological interpretation**: Consult malaria immunology literature
- **Method questions**: Refer to the scNET paper and documentation

## Citation

If you use this tutorial or adapt it for your research, please cite:
- The original scNET paper
- Any datasets you use
- This tutorial (if adapted significantly)

---

*This documentation addresses the request for malaria-associated B cell pathway reconstruction analysis. The tutorial provides a comprehensive workflow that users can adapt to their own datasets while maintaining the scientific rigor of the original paper's approach.*