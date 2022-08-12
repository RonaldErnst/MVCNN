import os


def main():
    # res no pool
    os.system(
        "python train_mvcnn.py -num_models 0 -weight_decay 0.001 -num_views 12 -dataset unified -bs 4 -num_epochs 15 -stage 2 -alterations nopool -svcnn_name i-uni-resnet-18-l1 -cnn_name resnet18_deep -svcnn_arc resnet18_deep -name i-uni-res-no-l2"
    )

    os.system(
        "python train_mvcnn.py -num_models 0 -weight_decay 0.001 -num_views 12 -dataset unified -bs 4 -num_epochs 1 -stage test -alterations nopool -svcnn_name i-uni-resnet-18-l1 -cnn_name resnet18_deep -svcnn_arc resnet18_deep -name i-uni-res-no-l2"
    )

    # res mean pool
    os.system(
        "python train_mvcnn.py -num_models 0 -weight_decay 0.001 -num_views 12 -dataset unified -bs 4 -num_epochs 15 -stage 2 -alterations mean -svcnn_name i-uni-resnet-18-l1 -cnn_name resnet18_deep -svcnn_arc resnet18_deep -name i-uni-res-mean-l2"
    )

    os.system(
        "python train_mvcnn.py -num_models 0 -weight_decay 0.001 -num_views 12 -dataset unified -bs 4 -num_epochs 1 -stage test -alterations mean -svcnn_name i-uni-resnet-18-l1 -cnn_name resnet18_deep -svcnn_arc resnet18_deep -name i-uni-res-mean-l2"
    )

    # res prepool pool
    os.system(
        "python train_mvcnn.py -num_models 0 -weight_decay 0.001 -num_views 12 -dataset unified -bs 4 -num_epochs 15 -stage 2 -alterations prepool -svcnn_name i-uni-resnet-18-l1 -cnn_name resnet18_deep -svcnn_arc resnet18_deep -name i-uni-res-pre-l2"
    )

    os.system(
        "python train_mvcnn.py -num_models 0 -weight_decay 0.001 -num_views 12 -dataset unified -bs 4 -num_epochs 1 -stage test -alterations prepool -svcnn_name i-uni-resnet-18-l1 -cnn_name resnet18_deep -svcnn_arc resnet18_deep -name i-uni-res-pre-l2"
    )

    # res postpool pool
    os.system(
        "python train_mvcnn.py -num_models 0 -weight_decay 0.001 -num_views 12 -dataset unified -bs 4 -num_epochs 15 -stage 2 -alterations postpool -svcnn_name i-uni-resnet-18-l1 -cnn_name resnet18_deep -svcnn_arc resnet18_deep -name i-uni-res-post-l2"
    )

    os.system(
        "python train_mvcnn.py -num_models 0 -weight_decay 0.001 -num_views 12 -dataset unified -bs 4 -num_epochs 1 -stage test -alterations postpool -svcnn_name i-uni-resnet-18-l1 -cnn_name resnet18_deep -svcnn_arc resnet18_deep -name i-uni-res-post-l2"
    )


if __name__ == "__main__":
    main()
