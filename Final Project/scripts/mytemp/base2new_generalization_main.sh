GPU=$1
SHOT=16
EPOCH=20

for dataset in eurosat dtd fgvc_aircraft oxford_flowers stanford_cars oxford_pets food101 sun397 ucf101 caltech101
do
    for seed in 1 2 3
    do
        for cfg in main
        do
        # training
            echo "train"
            sh scripts/mytemp/base2new_train.sh ${dataset} ${seed} ${GPU} ${cfg} ${SHOT}
        # evaluation
            echo "test"
            sh scripts/mytemp/base2new_test.sh ${dataset} ${seed} ${GPU} ${cfg} ${SHOT} ${EPOCH} base
            sh scripts/mytemp/base2new_test.sh ${dataset} ${seed} ${GPU} ${cfg} ${SHOT} ${EPOCH} new
        done
    done
done
