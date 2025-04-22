    #!/bin/bash

    # Define the datasets to run 'amazon' 'digit5' 'fashion_mnist')
    datasets=('cifar10' 'fashion_mnist') 
    # Define the algorithms to run 'FedAvg' 'BayesFedAvg' 'Wecfl' 'Fesem' 'GNN' 'JPDA' 'MHT'
    model_names=('IFCA')
    warm_ups=('False') # 'True'
    # Loop through the algorithms and run them with the defined parameters
    for dataset in "${datasets[@]}"
    do
        for model_name in "${model_names[@]}"
        do
            for K in 4
            do
                for warm_up in "${warm_ups[@]}"
                do
                    for n_assign in 3
                    do
                        for client_group in 10
                        do
                            sbatch main.bash $dataset $model_name $K $n_assign $warm_up $client_group
                        done
                    done
                done
            done
        done
    done