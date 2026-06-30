#!/usr/bin/env python3
"""
Main program interface for CA Paper SERS analysis pipeline.
This script serves as the main entry point for the SERS analysis pipeline.
"""
import argparse
import sys
import os
from pathlib import Path
from src.ca_paper_regression import CA_Paper_Unmixing_Models
from src.ca_paper_plsr import CA_Paper_PLSR_Unmixing
from src.byol_model import BYOLFullPipeline, CA_Paper_Full_Pipeline

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    """Main function to orchestrate the SERS analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="CA Paper SERS Analysis Pipeline - Main Program Interface"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ca_paper_plsr_unmixing", "ca_paper_unmixing_models",
                 "CA_Paper_Full_Pipeline", "byol_pipeline"],
        default="train",
        help="Operation mode: ca_paper_plsr_unmixing, ca_paper_unmixing_models, CA_Paper_Full_Pipeline, byol_pipeline"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data_MPAU_mix",
        help="Folder of the preprocessed data in ./data"
    )
    
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Path to model directory"
    )

    parser.add_argument(
        "--re_training",
        type=str,
        choices=["True", "False"],
        default="False",
        help="Retrain the model from checkpoint or not (True/False)"
    )


    args = parser.parse_args()
    
    print("=" * 60)
    print("Catecholamine SERS Analysis Pipeline - Main Program Interface")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Data directory: data/{args.data_dir}")
    print(f"Model directory: {args.model_dir}")
    print(f"Log directory: logs")
    print(f"visualization directory: visualizations")
    print("=" * 60)
    
    if args.mode == "ca_paper_unmixing_models":
        print("\nCA Paper Unmixing Models mode selected.")
        print("PLSR/RF/SVR/CNN component unmixing on selected data.")
        data_dir = os.path.join('data', args.data_dir)
        CA_Paper_Unmixing_Models(
            data_dir, args.model_dir,
            methods=['plsr', 'rf', 'svr', 'cnn'],
            random_state=2026, plot=True,
            mix_only=False, present_conc_range=(10, 20))
        print("CA Paper Unmixing Models completed.")

    elif args.mode == "ca_paper_plsr_unmixing":
        print("\nCA Paper PLSR Unmixing mode selected.")
        print("Plain holdout PLSR unmixing on selected data.")
        data_dir = os.path.join('data', args.data_dir)
        mix_filter = {'mix_only': False, 'present_conc_range': (10, 20)}
        CA_Paper_PLSR_Unmixing(data_dir, args.model_dir, plot=True, **mix_filter)
        print("CA Paper PLSR Unmixing completed.")

    elif args.mode == "byol_pipeline":
        print("\nBYOL Pre-training + Fine-tuning Pipeline selected.")
        data_dir = os.path.join('data', args.data_dir)
        mix_filter = {'mix_only': False, 'present_conc_range': (10, 20)}
        re_train = args.re_training == "True"
        BYOLFullPipeline(data_dir, args.model_dir,
                          conc_threshold=None,
                          **mix_filter,
                          stage1=True, stage2=True, re_training=re_train,
                          plot=True, dataset=args.data_dir)
        print("BYOL Pipeline completed.")

    elif args.mode == "CA_Paper_Full_Pipeline":
        print("\nCA Paper Full Pipeline mode selected.")
        print("BYOL classifier -> predicted-class PLSR quantification.")
        data_dir = os.path.join('data', args.data_dir)
        mix_filter = {'mix_only': False, 'present_conc_range': (10, 20)}
        re_train = args.re_training == "True"
        CA_Paper_Full_Pipeline(
            data_dir, args.model_dir, **mix_filter, plot=True, 
            re_training=re_train, stage1=True, dataset = args.data_dir)
        print("CA Paper Full Pipeline completed.")

    print("\nDone!")
    print("=" * 60)
    print("model saved in:", args.model_dir)
    print("visualization saved in: visualizations")
    print("log saved in: logs")


if __name__ == "__main__":
    main()
