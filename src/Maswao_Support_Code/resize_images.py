import os
import time
import json
from PIL import Image
from pathlib import Path
from datetime import datetime

class ImageResizeMonitor:
    def __init__(self, input_root, output_root, target_size=(227, 227), 
                 check_interval=4*3600, state_file='resize_state.json'):
        """
        Monitor input folder and resize new images as they appear.
        
        Args:
            input_root: Path to source folder being populated by slow script
            output_root: Path where resized images will be saved
            target_size: Tuple of (width, height) for resized images
            check_interval: Time in seconds between checks (default: 4 hours)
            state_file: JSON file to track processed images
        """
        self.input_path = Path(input_root)
        self.output_path = Path(output_root)
        self.target_size = target_size
        self.check_interval = check_interval
        self.state_file = Path(state_file)
        self.img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        
        # Load or initialize processed files state
        self.processed_files = self.load_state()
    
    def load_state(self):
        """Load the set of already processed files from state file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    print(f"Loaded state: {len(data)} files already processed")
                    return set(data)
            except Exception as e:
                print(f"Error loading state file: {e}")
                return set()
        return set()
    
    def save_state(self):
        """Save the set of processed files to state file."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(list(self.processed_files), f)
        except Exception as e:
            print(f"Error saving state file: {e}")
    
    def get_subfolder_status(self, subfolder_path):
        """
        Check if all images in a subfolder have been processed.
        Returns: (total_images, processed_images, needs_processing)
        """
        total = 0
        processed = 0
        
        if not subfolder_path.exists():
            return 0, 0, False
        
        for file in subfolder_path.iterdir():
            if file.is_file() and file.suffix.lower() in self.img_extensions:
                total += 1
                relative_path = str(file.relative_to(self.input_path))
                if relative_path in self.processed_files:
                    processed += 1
        
        needs_processing = processed < total
        return total, processed, needs_processing
    
    def process_subfolder(self, subfolder_path):
        """Process all unprocessed images in a single subfolder."""
        processed = 0
        errors = 0
        
        for file in subfolder_path.iterdir():
            if file.is_file() and file.suffix.lower() in self.img_extensions:
                relative_path = str(file.relative_to(self.input_path))
                
                # Skip if already processed
                if relative_path in self.processed_files:
                    continue
                
                if self.resize_image(file):
                    processed += 1
                else:
                    errors += 1
        
        return processed, errors
    
    def resize_image(self, input_file):
        """Resize a single image and save to output directory, maintaining folder structure."""
        try:
            # Construct output path maintaining full structure (train/dev/test/subfolder/image)
            relative_path = input_file.relative_to(self.input_path)
            output_file = self.output_path / relative_path
            
            # Create full output directory structure if needed
            # This ensures train/subfolder, dev/subfolder, test/subfolder are all created
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Resize and save
            with Image.open(input_file) as img:
                resized_img = img.resize(self.target_size, Image.LANCZOS)
                resized_img.save(output_file)
            
            # Mark as processed
            self.processed_files.add(str(relative_path))
            return True
            
        except Exception as e:
            print(f"Error processing {input_file}: {str(e)}")
            return False
    
    def process_new_images(self):
        """Process images subfolder by subfolder to avoid memory issues."""
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking for new images...")
        
        total_processed = 0
        total_errors = 0
        subfolders_processed = 0
        
        # Process each split (train/dev/test)
        for split in ['train', 'dev', 'test']:
            split_path = self.input_path / split
            
            if not split_path.exists():
                continue
            
            print(f"\nProcessing {split}/ folder...")
            
            # Get all subfolders in this split
            subfolders = [d for d in split_path.iterdir() if d.is_dir()]
            
            for subfolder in subfolders:
                # Check status of this subfolder
                total, processed_count, needs_processing = self.get_subfolder_status(subfolder)
                
                if not needs_processing:
                    # All images in this subfolder already processed
                    continue
                
                subfolder_name = subfolder.relative_to(self.input_path)
                print(f"  Processing {subfolder_name}: {processed_count}/{total} already done")
                
                # Process only unprocessed images in this subfolder
                processed, errors = self.process_subfolder(subfolder)
                
                total_processed += processed
                total_errors += errors
                subfolders_processed += 1
                
                # Save state after each subfolder
                if processed > 0:
                    self.save_state()
                    print(f"    Processed {processed} new images from {subfolder_name}")
        
        if subfolders_processed == 0:
            print("No new images found in any subfolder.")
        else:
            print(f"\nBatch complete: {total_processed} images processed, {total_errors} errors")
            print(f"Processed {subfolders_processed} subfolders")
        
        return total_processed
    
    def run(self, max_iterations=None):
        """
        Run the monitor continuously.
        
        Args:
            max_iterations: Maximum number of check cycles (None for infinite)
        """
        print("="*60)
        print("Image Resize Monitor Started")
        print(f"Input folder: {self.input_path}")
        print(f"Output folder: {self.output_path}")
        print(f"Target size: {self.target_size}")
        print(f"Check interval: {self.check_interval/3600:.1f} hours")
        print(f"Monitoring: train/, dev/, and test/ folders")
        print("="*60)
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                
                # Process new images
                self.process_new_images()
                
                # Check if we should stop
                if max_iterations and iteration >= max_iterations:
                    print("\nReached maximum iterations. Stopping.")
                    break
                
                # Wait before next check
                print(f"\nWaiting {self.check_interval/3600:.1f} hours until next check...")
                print(f"Next check at: {datetime.fromtimestamp(time.time() + self.check_interval).strftime('%Y-%m-%d %H:%M:%S')}")
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            print("\n\nMonitor stopped by user.")
            self.save_state()
            print(f"Total images processed: {len(self.processed_files)}")


# Example usage:
if __name__ == "__main__":
    # Configuration
    input_root = "../../../M2M-MT-files/Youtube_Dipolelo_Video_Frames"
    output_root = "../../../M2M-MT-files/Youtube_Dipolelo_Video_Frames_Resized"
    
    # Create monitor instance
    monitor = ImageResizeMonitor(
        input_root=input_root,
        output_root=output_root,
        target_size=(227, 227),
        check_interval=4*3600,  # 4 hours in seconds (change to 5*3600 for 5 hours)
        state_file='resize_state.json'  # Tracks what's been processed
    )
    
    # Run indefinitely (will check every 4 hours)
    # Press Ctrl+C to stop
    monitor.run()
    
    # Or run for a specific number of iterations:
    # monitor.run(max_iterations=10)  # Check 10 times then stop