#!/usr/bin/env python3
"""
Main program interface for CA Paper SERS analysis pipeline.
This script serves as the main entry point for the SERS analysis pipeline.
"""
import argparse
import sys
import os
from pathlib import Path
from src.train import train_model
from src.predict import test_Identification_Model
from src.train import test_train_ratio_model
from src.predict import test_Ratio_Model
from src.predict import Ratio_prediction_test
from src.plsr_model import PLSR_Train, PLSR_Predict
from src.ca_paper_plsr import CA_Paper_PLSR
from src.ca_paper_regression import CA_Paper_Regression, CA_Paper_Unmixing_Models
from src.ca_paper_plsr import CA_Paper_PLSR_Unmixing
from src.teacher_student_model import TeacherStudentTrain, TeacherStudentPLSRQuant
from src.nnls_unmixing import NNLSUnmixing
from src.dino_model import BYOLFullPipeline, CA_Paper_Full_Pipeline
from src.utils import read_spectra_train, read_spectra_test, spectra_normalization
import numpy as np

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
        help="Operation mode: train, predict, test, plsr_train, or plsr_test"
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
        "--task",
        type=str,
        choices=["classification", "quantification"],
        default="classification",
        help="Task for teacher_student mode: classification or quantification"
    )

    parser.add_argument(
        "--re_training",
        type=str,
        choices=["True", "False"],
        default="False",
        help="Resume from checkpoint (True) or train from scratch (False)"
    )

    parser.add_argument(
        "--stage2_task",
        type=str,
        choices=["classification", "quantification_plsr", "quantification_mlp"],
        default="classification",
        help="Stage 2 task for byol_pipeline mode"
    )

    args = parser.parse_args()
    
    print("=" * 60)
    print("Transfer Learning Assisted SERS")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Data directory: {args.data_dir}")
    print(f"Model directory: {args.model_dir}")
    print("=" * 60)
    
    if args.mode == "train":
        print("\nTraining mode selected.")
        train_model(os.path.join(args.data_dir, 'train'), args.model_dir)

        
    elif args.mode == "predict":
        print("\nPrediction mode selected.")
        print("Prediction functionality to be implemented in src/ directory")
        # TODO: Import and call prediction function from src/
    
    elif args.mode == "test":
        print("\nTesting mode selected.")
        Ratio_prediction_test(os.path.join(args.data_dir, 'test'), args.model_dir)

    elif args.mode == "test_IdModel":
        print("\nTesting Identification Model mode selected.")
        test_Identification_Model(os.path.join(args.data_dir, 'test'), args.model_dir)

    elif args.mode == "test_RatioModel_train":
        print("\nTesting Ratio Model on training data mode selected.")
        test_train_ratio_model(os.path.join(args.data_dir, 'train'), args.model_dir)

    elif args.mode == "test_RatioModel_predict":
        # the test is on train set
        print("\nTesting Ratio Model on prediction data mode selected.")
        test_Ratio_Model(os.path.join(args.data_dir, 'train'), args.model_dir)

    elif args.mode == "plsr_train":
        print("\nPLSR Training mode selected.")
        Raman_Shift, Intensity, Category, Concentration, _ = read_spectra_train(
            os.path.join(args.data_dir, 'train')
        )
        print(f"Raman Shift shape: {Raman_Shift.shape}")
        print(f"Intensity shape: {Intensity.shape}")
        Intensity_norm = spectra_normalization(
            Raman_Shift, Intensity,
            peak_position=920, peak_range=20, plot=True, mode='plsr_train', minmax_scale = False
        )
        print("Data normalization completed.")
        PLSR_Train(Raman_Shift, Intensity_norm, Concentration, Category,
                   args.model_dir, plot=True)
        print("PLSR model trained successfully.")

    elif args.mode == "plsr_test":
        print("\nPLSR Testing mode selected.")
        Raman_Shift, Intensity, Concentrations, Groups = read_spectra_test(
            os.path.join(args.data_dir, 'test')
        )
        print(f"Raman Shift shape: {Raman_Shift.shape}")
        print(f"Intensity shape: {Intensity.shape}")
        Intensity_norm = spectra_normalization(
            Raman_Shift, Intensity,
            peak_position=920, peak_range=20, plot=True, mode='plsr_test', minmax_scale = False
        )
        print("Data normalization completed.")
        total_true = np.sum(Concentrations, axis=1)
        predictions = PLSR_Predict(Intensity_norm, args.model_dir,
                                   plot=True, true_concentrations=total_true,
                                   groups=Groups)
        rmse = np.sqrt(np.mean((predictions - total_true) ** 2))
        print(f"PLSR Test Results:")
        print(f"  RMSE: {rmse:.4f} uM")
        unique_folders = np.unique(Groups)
        name_width = max(len(f) for f in unique_folders)
        for folder in unique_folders:
            mask = Groups == folder
            true_val = total_true[mask][0]
            pred_mean = np.mean(predictions[mask])
            pred_std = np.std(predictions[mask])
            print(f"  {folder:<{name_width}}  "
                  f"True={true_val:6.2f} uM  "
                  f"Pred={pred_mean:6.2f} ± {pred_std:.2f} uM")

    elif args.mode == "ca_paper_plsr":
        print("\nCA Paper PLSR mode selected.")
        print("Using concentration-level train/test split (no within-batch leakage).")
        data_water_dir = os.path.join(args.data_dir, 'data_water')
        CA_Paper_PLSR(data_water_dir, args.model_dir,
                      train_ratio=0.7, random_state=42, plot=True)
        print("CA Paper PLSR completed.")

    elif args.mode == "ca_paper_regression":
        print("\nCA Paper Regression mode selected.")
        print("Running RF, SVR, CNN (random mixing) comparison on data_water.")
        data_water_dir = os.path.join(args.data_dir, 'data_MPAU')
        CA_Paper_Regression(data_water_dir, args.model_dir,
                           methods=['plsr', 'rf', 'svr', 'cnn'],
                           train_ratio=0.7, random_state=42, plot=True)
        print("CA Paper Regression completed.")

    elif args.mode == "ca_paper_unmixing_models":
        print("\nCA Paper Unmixing Models mode selected.")
        print("PLSR/RF/SVR/CNN component unmixing on data_MPAU2_mix.")
        data_mpau_dir = os.path.join(args.data_dir, 'data_MPAU2_mix')
        CA_Paper_Unmixing_Models(
            data_mpau_dir, args.model_dir,
            methods=['plsr', 'rf', 'svr', 'cnn'],
            random_state=2026, plot=True,
            mix_only=False, present_conc_range=(10, 20))
        print("CA Paper Unmixing Models completed.")

    elif args.mode == "ca_paper_plsr_unmixing":
        print("\nCA Paper PLSR Unmixing mode selected.")
        print("Plain holdout PLSR unmixing on data_MPAU2_mix.")
        data_mpau_dir = os.path.join(args.data_dir, 'data_MPAU2_mix')
        mix_filter = {'mix_only': False, 'present_conc_range': (10, 20)}
        CA_Paper_PLSR_Unmixing(data_mpau_dir, args.model_dir, plot=True,
                                **mix_filter)
        print("CA Paper PLSR Unmixing completed.")

    elif args.mode == "CA_Paper_Full_Pipeline":
        print("\nCA Paper Full Pipeline mode selected.")
        print("BYOL classifier -> predicted-class PLSR quantification.")
        dataset_name = 'MPAU2'
        data_mpau_dir = os.path.join(args.data_dir, f'data_{dataset_name}_mix')
        threshold_map = {'MPAU': None, 'Water': None, 'MPAU2': None}
        mix_filter = {'mix_only': False, 'present_conc_range': (10, 20)}
        re_train = args.re_training == "True"
        CA_Paper_Full_Pipeline(
            data_mpau_dir, args.model_dir,
            conc_threshold=threshold_map[dataset_name],
            **mix_filter,
            plot=True, dataset=dataset_name,
            re_training=re_train, stage1=True)
        print("CA Paper Full Pipeline completed.")

    elif args.mode == "teacher_student":
        print(f"\nTeacher-Student Model mode selected (task={args.task}).")
        dataset_name = 'MPAU2'
        data_mpau_dir = os.path.join(args.data_dir, f'data_{dataset_name}_mix')
        # conc_threshold: exclude groups with total > threshold (uM) to avoid
        # adsorption saturation effects. MPAU=45, Water=None (adjust as needed)
        threshold_map = {'MPAU': None, 'Water': None, 'MPAU2': None}
        # mix_only: keep binary+ternary only; present_conc_range: (min, max)
        # for each present component. None=no filter, (10,20)=only 10-20 uM.
        mix_filter = {'mix_only': False, 'present_conc_range': (0, 20)}
        TeacherStudentTrain(data_mpau_dir, args.model_dir,
                            task=args.task,
                            epochs=300, batch_size=16, lr=1e-3,
                            random_state=2026, loss_w_total = 3.0,
                            conc_threshold=threshold_map[dataset_name],
                            **mix_filter,
                            plot=True, dataset = dataset_name, train = True)
        print(f"Teacher-Student {args.task} completed.")

    elif args.mode == "teacher_student_plsr":
        print(f"\nTeacher-Student + PLSR OOF mode selected.")
        dataset_name = 'MPAU2'
        data_mpau_dir = os.path.join(args.data_dir, f'data_{dataset_name}_mix')
        print(f"  Dataset: {dataset_name}")
        print(f"  Stage 1: train encoder via classification")
        print(f"  Stage 2: 3-fold OOF PLSR on frozen 256-dim features")
        threshold_map = {'MPAU': None, 'Water': None, 'MPAU2': None}
        mix_filter = {'mix_only': True, 'present_conc_range': (10, 20)}
        TeacherStudentPLSRQuant(
            data_mpau_dir, args.model_dir,
            conc_threshold=threshold_map[dataset_name],
            **mix_filter,
            train_encoder=True, epochs=500, lr=1e-3,
            random_state=2026, dataset=dataset_name, plot=True)
        print("Teacher-Student PLSR OOF completed.")

    elif args.mode == "nnls_unmixing":
        print(f"\nRegularized NNLS-PLSR unmixing mode selected.")
        dataset_name = 'MPAU2'
        data_mpau_dir = os.path.join(args.data_dir, f'data_{dataset_name}_mix')
        print(f" Dataset: {dataset_name}")
        threshold_map = {'MPAU': None, 'Water': None, 'MPAU2': None}
        NNLSUnmixing(data_mpau_dir, args.model_dir,
                      conc_threshold=threshold_map[dataset_name],
                      random_state=2026, plot=True)
        print("Regularized NNLS-PLSR unmixing completed.")

    elif args.mode == "byol_pipeline":
        print("\nBYOL Pre-training + Fine-tuning Pipeline selected.")
        dataset_name = 'MPAU2'
        data_mpau_dir = os.path.join(args.data_dir, f'data_{dataset_name}_mix')
        threshold_map = {'MPAU': None, 'Water': None, 'MPAU2': None}
        mix_filter = {'mix_only': False, 'present_conc_range': (10, 20)}
        re_train = args.re_training == "True"
        BYOLFullPipeline(data_mpau_dir, args.model_dir,
                          conc_threshold=threshold_map[dataset_name],
                          **mix_filter,
                          stage1=True, stage2=True, re_training=re_train,
                          stage2_task=args.stage2_task,
                          plot=True, dataset=dataset_name)
        print("BYOL Pipeline completed.")

    print("\nDone!")


if __name__ == "__main__":
    main()
