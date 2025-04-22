    #!/bin/bash

    # Define the datasets to run 'amazon' 'digit5' 'digit5')
    datasets=('digit5') 
    # Define the algorithms to run 'FedAvg' 'BayesFedAvg' 'Wecfl' 'Fesem' 'GNN' 'JPDA' 'MHT'
    model_names=('Wecfl')
    warm_ups=('False')
    # Loop through the algorithms and run them with the defined parameters
    for dataset in "${datasets[@]}"
    do
        for model_name in "${model_names[@]}"
        do
            for K in 7 8
            do
                for warm_up in "${warm_ups[@]}"
                do
                    for n_assign in 3 #6
                    do
                        for client_group in 2
                        do
                            sbatch main_cpu.bash $dataset $model_name $K $n_assign $warm_up $client_group
                        done
                    done
                done
            done
        done
    done