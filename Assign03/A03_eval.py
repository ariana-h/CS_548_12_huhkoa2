import torch_fidelity as torf

metrics = torf.calculate_metrics(
    input1="gen_images/",
    input2="train_images/",
    fid=True,
    kid=True,
    kid_subset_size=700,
    cuda=True)
