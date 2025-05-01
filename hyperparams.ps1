$layers = @(2, 4, 6, 8, 10, 12)
$hidden_size = @(64, 128, 256, 512)
$learning_rate = @(0.001, 0.01, 0.1, 0.005)
$max_epochs = @(1000, 5000, 10000, 20000)

foreach ($layer in $layers) {
    foreach ($size in $hidden_size) {
        foreach ($rate in $learning_rate) {
            foreach ($epoch in $max_epochs) {
                Write-Host "Training with parameters: Layers=$layer, Hidden Size=$size, Learning Rate=$rate, Max Epochs=$epoch"
                python main.py -te train --run RNN2 --max_epochs $epoch --lr $rate --num_layers $layer --hidden_size $size
            }
        }
    }
}
