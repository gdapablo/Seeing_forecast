import argparse
import os
from src.preprocess.wht_cleaner import WHTCleaner
from src.preprocess.int_cleaner import INTCleaner
from src.preprocess.seeing_cleaner import SeeingCleaner
from src.visualizer import WeatherVisualizer, SeeingVisualizer
from src.ml_modeling import MLModelingPipeline

def main():
    parser = argparse.ArgumentParser(description="Seeing Forecast Pipeline")
    parser.add_argument('--task', type=str, required=True,
                        choices=['visualize', 'ml'], help="Task to perform")
    parser.add_argument('--dataset', type=str,
                        choices=['WHT', 'INT', 'seeing'], help="Dataset to process or visualize")
    parser.add_argument('--visual_target', type=str,
                        choices=['monthly_means', 'yearly_means', 'seeing_month', 'seeing_variation'],
                        help="Visualization target")
    parser.add_argument('--model', type=str,
                        choices=['random_forest', 'polynomial', 'dnn'], help="ML model to train and evaluate")
    parser.add_argument('--save', action='store_true', help="Save the plots to disk")

    args = parser.parse_args()

    if args.task == 'ml':
        if args.model is None:
            print("Specify the model with --model")
            return

        cleaner = INTCleaner(folder_path='data/INT_data')
        df_int = cleaner.load_data()
        df_clean = cleaner.preprocess()

        cleaner_seeing = SeeingCleaner(filepath='data/seeing_data.csv')
        seeing = cleaner_seeing.load_and_preprocess()

        print(f"Running ML model: {args.model} ...")
        pipeline = MLModelingPipeline()
        pipeline.run_model(df_clean, seeing, args.model)

    elif args.task == 'visualize':
        if args.visual_target is None:
            print("Specify --visual_target for visualization.")
            return

        if args.visual_target in ['monthly_means', 'yearly_means']:
            cleaner = INTCleaner(folder_path='data/INT_data')
            df_int = cleaner.load_data()
            df_clean = cleaner.preprocess()
            visualizer = WeatherVisualizer()

            if args.visual_target == 'monthly_means':
                visualizer.plot_monthly_means(data=df_int, save=args.save)
            elif args.visual_target == 'yearly_means':
                visualizer.plot_yearly_means(data=df_int, save=args.save)

        elif args.visual_target == 'seeing_month':
            cleaner = SeeingCleaner(filepath='data/seeing_data.csv')
            df_seeing = cleaner.load_and_preprocess()
            visualizer = SeeingVisualizer()
            visualizer.plot_monthly_seeing(data=df_seeing, save=args.save)

        elif args.visual_target == 'seeing_variation':
            cleaner = SeeingCleaner(filepath='data/seeing_data.csv')
            df_seeing = cleaner.load_and_preprocess()
            visualizer = SeeingVisualizer()
            visualizer.plot_minute_seeing_variation(data=df_seeing, save=args.save)

        else:
            print("Visual_target unknown.")
    else:
        print("Unknown task. Specify a valid one --task clean or visualize.")

if __name__ == '__main__':
    main()
