python train_recon_sscvae_pretrain.py -c configs/sscvae_recon_sevir_pretrain.json
python scripts/filter_model.py -c configs/sscvae_recon_sevir_pretrain.json
python test_recon_sscvae_pretrain.py -c configs/sscvae_recon_sevir_pretrain.json
python train_recon_sscvae_pretrain.py -c configs/sscvae_recon_sevir_pretrain_small.json
python test_recon_sscvae_pretrain.py -c configs/sscvae_recon_sevir_pretrain_small.json

python train_recon_sscvae_trans.py -c configs/sscvae_recon_sevir_trans.json
python scripts/filter_model.py -c configs/sscvae_recon_sevir_trans.json
python test_recon_sscvae_trans.py -c configs/sscvae_recon_sevir_trans.json