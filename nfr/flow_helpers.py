import nflows.transforms
from nflows.transforms.base import CompositeTransform


def create_linear_transform(options):
    if options["linear_transform_type"] == "reverse":
        return nflows.transforms.ReversePermutation(features=options["D"])
    else:
        raise NotImplementedError(
            f"Unknown linear transform type: {options['linear_transform_type']}. Only 'reverse' is supported."
        )


def create_base_transform(i, options):
    if options["base_transform_type"] == "affine-autoregressive":
        return nflows.transforms.MaskedAffineAutoregressiveTransform(
            features=options["D"],
            hidden_features=options["hidden_features"],
            context_features=None,
            num_blocks=options["num_transform_blocks"],
            use_residual_blocks=True,
            random_mask=False,
            activation=options["activation_fn"],
            dropout_probability=options["dropout_probability"],
            use_batch_norm=options["use_batch_norm"],
            constrain_transform_ranges=options["constrain_transform_ranges"],
            transform_range_scales={
                k: options[k]
                for k in [
                    "inscale_alpha",
                    "outcale_alpha",
                    "inscale_mu",
                    "outscale_mu",
                ]
            },
            return_aux=options.get("return_aux", False),
        )
    else:
        raise ValueError(
            f"Unknown base transform type: {options['base_transform_type']}. Only 'affine-autoregressive' is supported."
        )


def create_transforms(options):
    all_transforms = []  # transforms for hidden intermidate flows
    for i in range(options["n_layers"] + 2):
        transform = []
        transform = CompositeTransform(
            [
                create_linear_transform(options),
                create_base_transform(i, options),
            ]
        )
        all_transforms.append(transform)
    return all_transforms
