
# conda activate tesi & cd Desktop\tesi\Oral_3d & python main.py --mode train 

from utils.dataset import Dataset
from model.solver import Solver
from model.oral_3d import Oral3D
import argparse

arg_parser = argparse.ArgumentParser(description="Training settings for Oral-3D")
arg_parser.add_argument('--test_mode', action="store_true", 
                        help='different test option')
arg_parser.add_argument('--data_root', dest='data_root', type=str, default='./data/mat_oral_3d',
                        help='location of training/val/test data and split info')
arg_parser.add_argument('--device', dest='device', type=str, default='cuda:0',
                        help='device to run the code')
arg_parser.add_argument('--train_n', dest='train_n', type=int, default=300, #300
                        help='epoch nums for training')
arg_parser.add_argument('--val_n', dest='val_n', type=int, default=20,#20
                        help='ogni quante epoche il modello deve essere validato')
arg_parser.add_argument('--save_n', dest='save_n', type=int, default=20,#20
                        help=' ogni quante epoche il modello deve essere salvato')
arg_parser.add_argument('--d_start_n', dest='d_start_n', type=int, default=100,#100
                        help='epoch nums to start training discriminator')
arg_parser.add_argument('--g_lr', dest='g_lr', type=float, default=0.001,
                        help='learning rate for generator/discriminator')
arg_parser.add_argument('--d_lr', dest='d_lr', type=float, default=0.001,
                        help='learning rate for generator/discriminator')
arg_parser.add_argument('--mode', dest='mode', type=str, default='test',
                        help='train/test/infer model')
arg_parser.add_argument('--mat_path',  dest='mat_path',type=str, default='./png_images', help='Percorso immagine panoramica PNG')
#arg_parser.add_argument('--curve_dir',  dest='curve_dir',type=str, default='./curve',help='Percorso file curva dentale')
arg_parser.add_argument('--resume', type=int, 
                       help='Percorso al checkpoint per riprendere il training')
arg_parser.add_argument('--save_root', type=str, default='./output',
                        help='Percorso base per salvare i risultati')
arg_parser.add_argument('--output_dir',dest='output_dir', type=str, default='./output/mat', 
                       help='Cartella di output per le ricostruzioni')
arg_parser.add_argument('--test_only', action='store_true',
                        help='whether loading test data only')
arg_parser.add_argument('--lr_scheduler', dest='lr_scheduler', type=str, default='manual',choices=['manual', 'plateau','cosine'],
                       help='Metodo di gestione del learning rate: manual, plateau o cosine')
arg_parser.add_argument('--early_stopping', type=int, default=0,
                        help='Abilita early stopping (1=attivo, 0=disattivo)')


if __name__ == '__main__':
    args = arg_parser.parse_args()
    
    # Solo per modalità train/test
    dataset = Dataset(args.data_root, test_only=args.test_only) if args.mode in ['train', 'test'] else None
    
    oral_3d_model = Oral3D(device=args.device)
    solver = Solver(dataset, oral_3d_model, MPR=False, args=args)

    if args.mode == 'train':
        if args.resume:
            solver._load_ckpt(args.resume)
        solver.train()

    elif args.mode == 'test':
        solver.test()
    elif args.mode == 'infer':
        # Passa esplicitamente i parametri
        solver.infer(
            mat_path=args.mat_path,
            output_dir=args.output_dir
        )


        


        #  python main.py --mode train 
        # python main.py --mode test --test_only 

        # python main.py --mode infer --png_dir ./png_images --curve_dir ./curve
