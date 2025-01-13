import allel, argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcf", help="path to VCF (or .vcf.gz)")
    parser.add_argument("--zarr", help="path for zarr output")
    parser.add_argument(
        "--overwrite", action="store_true", help="overwrite existing zarr file"
    )
    args = parser.parse_args()

    allel.vcf_to_zarr(args.vcf, args.zarr, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    main()
