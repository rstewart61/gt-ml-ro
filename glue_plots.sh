#!/bin/bash

montage -mode concatenate tuning_plots/MIMIC* tuning_plots/GA* tuning_plots/SA* tuning_plots/RHC* -tile 3x4 tuning_plots/montage.png

montage -mode concatenate tuning_plots/nn/Polish* -tile 3x6 tuning_plots/nn/montage.png


for problem in "Continuous Peaks" "Four Peaks" Knapsack "Max K-Color" "Traveling Sales"; do
    LEFT=plots/"${problem}_Fitness by Function Calls*.png"
    RIGHT=plots/"${problem}_Fitness by Iterations*.png"
    echo "$problem"
    echo "$LEFT" "$RIGHT"
    convert +append "$LEFT" "$RIGHT" plots/"$problem"_"Fitness Combined.png"
    LEFT=plots/"${problem}_Fitness by length*.png"
    RIGHT=plots/"${problem}_Function Calls by length*.png"
    echo "$LEFT" "$RIGHT"
    convert +append "$LEFT" "$RIGHT" plots/"$problem"_"Function Calls Combined.png"
done

LEFT=tuning_plots/"Max K-Color_MIMIC_pop_size.png"
RIGHT=tuning_plots/"Traveling Sales_MIMIC_pop_size.png"
convert +append "$LEFT" "$RIGHT" tuning_plots/"_MIMIC_pop_size_Traveling Sales vs Max K-Color.png"

LEFT=plots/nn/"Polish Bankruptcy_bar.png"
RIGHT=plots/nn/"Polish Bankruptcy_iterations.png"
convert +append "$LEFT" "$RIGHT" plots/nn/"Polish Bankruptcy_combined.png"

convert +append plots/*convergence_times.png plots/convergence_times_combined.png

LEFT=tuning_plots/nn/"Polish Bankruptcy_RHC_restarts.png"
RIGHT=tuning_plots/nn/"Polish Bankruptcy_RHC_max_attempts.png"
convert +append "$LEFT" "$RIGHT" tuning_plots/nn/"_Polish Bankruptcy_RHC_restarts_and_max_attempts.png"

