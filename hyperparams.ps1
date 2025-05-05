$layers = @(2, 4, 8, 16)
$hidden_size = @(64, 128, 256)
$learning_rate = @(0.001, 0.01, 0.1)
$max_epochs = @(10, 20, 30)


foreach ($layer in $layers) {
    foreach ($size in $hidden_size) {
        foreach ($rate in $learning_rate) {
            foreach ($epoch in $max_epochs) {
                Write-Host "ℹ️ Training with parameters: Layers=$layer, Hidden Size=$size, Learning Rate=$rate, Max Epochs=$epoch"
                # $params = "--num_layers $layer --hidden_size $size --learning_rate $rate --max_epochs $epoch"
                # Example command "python nameGeneration.py --num_layers 2 --hidden_size 64 --learning_rate 0.001 --max_epochs 1000"
                .\passwordGenEnv\Scripts\python.exe main.py --num_layers $layer --hidden_size $size --learning_rate $rate --max_epochs $epoch
            }
        }
    }
}