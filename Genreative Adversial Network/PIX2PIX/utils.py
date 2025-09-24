import os
import torch
import torchvision.utils as vutils



def save_generated_images(generator, val_loader, epoch, device, out_dir, num_samples=4):
    generator.eval()
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (x, y) in enumerate(val_loader):
            x = x.to(device)
            y = y.to(device)
            y_fake = generator(x)

            # Save a grid: [input | fake | target]
            grid = torch.cat([x[:num_samples], y_fake[:num_samples], y[:num_samples]], dim=0)
            vutils.save_image(grid, os.path.join(out_dir, f"epoch_{epoch+1}.png"), nrow=num_samples, normalize=True)
            break  # only save first batch

    generator.train()

    print(f"Saved generated images at: {out_dir}/epoch_{epoch+1}.jpg")

    # # show inline (optional during training)
    # grid = torchvision.utils.make_grid(generated_imgs, nrow=4, normalize=True)
    # plt.figure(figsize=(6, 6))   # better scaling
    # plt.imshow(np.transpose(grid, (1, 2, 0)))
    # plt.title(f"Epoch {epoch+1}")
    # plt.axis('off')
    # plt.show()




def save_checkpoint(generator, discriminator, optimizer_G, optimizer_D, epoch, checkpoint_dir,max_keep=5):
    os.makedirs(checkpoint_dir, exist_ok=True)

    """Save model checkpoints and keep only the last 'max_keep' files."""
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")

    # Save both Generator & Discriminator states
    torch.save({
        'epoch': epoch+1,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
    }, checkpoint_path)

    # Manage only last max_keep checkpoints
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    # Sort numerically by epoch number
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    # Remove oldest if exceeding max_keep
    while len(checkpoints) > max_keep:
        os.remove(os.path.join(checkpoint_dir, checkpoints[0]))
        checkpoints.pop(0)

# Resume training if checkpoint exists
def load_latest_checkpoint(generator, discriminator, optimizer_G, optimizer_D, checkpoint_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(checkpoint_dir):
        return 0  # start from epoch 0

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if len(checkpoints) == 0:
        return 0  # no checkpoints found

    # Sort numerically by epoch number
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    latest_checkpoint = checkpoints[-1]  # now this is truly the latest epoch
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

    return checkpoint['epoch']  # return the last trained epoch
