import argparse
from src.preprocessing.wht_cleaner import WHTCleaner
from src.preprocessing.int_cleaner import INTCleaner
from src.preprocessing.seeing_cleaner import SeeingCleaner
from src.visualization.visualizer import WeatherVisualizer, SeeingVisualizer

def main():
    parser = argparse.ArgumentParser(description="Weather & Seeing Data Pipeline")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["clean_wht", "clean_int", "clean_seeing", "visualize"],
        help="Task to execute",
    )
    parser.add_argument(
        "--visual_target",
        type=str,
        choices=["weather_month", "weather_year", "seeing_month", "seeing_minute"],
        help="Target plot for visualization (used only with --task visualize)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save plots as PNG images instead of displaying them"
    )

    args = parser.parse_args()

    if args.task == "clean_wht":
        cleaner = WHTCleaner()
        df_wht = cleaner.run_cleaning_pipeline()
        print("WHT data cleaned and preprocessed.")

    elif args.task == "clean_int":
        cleaner = INTCleaner()
        df_int = cleaner.run_cleaning_pipeline()
        print("INT data cleaned and preprocessed.")

    elif args.task == "clean_seeing":
        cleaner = SeeingCleaner()
        df_seeing = cleaner.run_cleaning_pipeline()
        print("Seeing data cleaned and preprocessed.")

    elif args.task == "visualize":
        if not args.visual_target:
            print("⚠️ Please provide --visual_target when using --task visualize.")
            return

        if args.visual_target in ["weather_month", "weather_year"]:
            cleaner = INTCleaner()
            df_int = cleaner.run_cleaning_pipeline()
            visualizer = WeatherVisualizer(df_int)
            if args.visual_target == "weather_month":
                visualizer.plot_monthly_means(save=args.save)
            elif args.visual_target == "weather_year":
                visualizer.plot_yearly_means(save=args.save)

        elif args.visual_target in ["seeing_month", "seeing_minute"]:
            cleaner = SeeingCleaner()
            df_seeing = cleaner.run_cleaning_pipeline()
            visualizer = SeeingVisualizer(df_seeing)
            if args.visual_target == "seeing_month":
                visualizer.plot_monthly_seeing(save=args.save)
            elif args.visual_target == "seeing_minute":
                visualizer.plot_seeing_variation_by_minute(save=args.save)


if __name__ == "__main__":
    main()
