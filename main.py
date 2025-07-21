import argparse
import os
from src.preprocess.wht_cleaner import WHTCleaner
from src.preprocess.int_cleaner import INTCleaner
from src.preprocess.seeing_cleaner import SeeingCleaner
from src.visualizer import WeatherVisualizer, SeeingVisualizer

def main():
    parser = argparse.ArgumentParser(description="Seeing Forecast Pipeline")
    parser.add_argument('--task', type=str, required=True,
                        choices=['clean', 'visualize'], help="Task to perform")
    parser.add_argument('--dataset', type=str,
                        choices=['WHT', 'INT', 'seeing'], help="Dataset to process or visualize")
    parser.add_argument('--visual_target', type=str,
                        choices=['temp_month', 'temp_year', 'wind_month', 'wind_year', 'humidity_month', 'humidity_year', 'seeing_month', 'seeing_variation'],
                        help="Visualization target")
    parser.add_argument('--save', action='store_true', help="Save the plots to disk")

    args = parser.parse_args()

    if args.task == 'clean':
        if args.dataset == 'WHT':
            cleaner = WHTCleaner(folder_path='data/WHT_data')
            df_wht = cleaner.load_and_preprocess()
            print("WHT data cleaned and loaded.")
            # Guardar o procesar df_wht si es necesario
        elif args.dataset == 'INT':
            cleaner = INTCleaner(folder_path='data/INT_data')
            df_int = cleaner.load_and_preprocess()
            print("INT data cleaned and loaded.")
            # Guardar o procesar df_int si es necesario
        elif args.dataset == 'seeing':
            cleaner = SeeingCleaner(filepath='data/seeing_data.csv')
            df_seeing = cleaner.load_and_preprocess()
            print("Seeing data cleaned and loaded.")
            # Guardar o procesar df_seeing si es necesario
        else:
            print("Por favor especifica un dataset válido para limpieza: WHT, INT, o seeing.")

    elif args.task == 'visualize':
        if args.visual_target is None:
            print("Por favor especifica --visual_target para visualización.")
            return

        # Visualizaciones que requieren datos INT
        if args.visual_target in ['temp_month', 'temp_year', 'wind_month', 'wind_year', 'humidity_month', 'humidity_year']:
            cleaner = INTCleaner(folder_path='data/INT_data')
            df_int = cleaner.load_and_preprocess()
            visualizer = WeatherVisualizer()

            if args.visual_target == 'temp_month':
                visualizer.plot_mean_temperature_per_month(data=df_int, save=args.save)
            elif args.visual_target == 'temp_year':
                visualizer.plot_mean_temperature_per_year(data=df_int, save=args.save)
            elif args.visual_target == 'wind_month':
                visualizer.plot_mean_wind_per_month(data=df_int, save=args.save)
            elif args.visual_target == 'wind_year':
                visualizer.plot_mean_wind_per_year(data=df_int, save=args.save)
            elif args.visual_target == 'humidity_month':
                visualizer.plot_mean_humidity_per_month(data=df_int, save=args.save)
            elif args.visual_target == 'humidity_year':
                visualizer.plot_mean_humidity_per_year(data=df_int, save=args.save)

        # Visualizaciones del seeing
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
            print("Visual_target desconocido. Por favor especifica una opción válida.")
    else:
        print("Tarea desconocida. Por favor especifica --task clean o visualize.")

if __name__ == '__main__':
    main()
