# main.tf - Terraform configuration for Kalaa-Setu GCP GPU Instance

# Configure the Google Cloud provider
provider "google" {
  project = "hpe-managed-services" # <-- IMPORTANT: Replace with your GCP Project ID
  zone    = "us-central1-a"       # You can change this to a zone closer to you
}

# Define the GCP Compute Engine instance for Kalaa-Setu
resource "google_compute_instance" "kalaasetu_vm" {
  name         = "kalaasetu-demo-vm"
  machine_type = "n1-standard-4" # 4 vCPUs, 15 GB RAM

  # Use a pre-built Deep Learning image with CUDA and drivers installed
  boot_disk {
    initialize_params {
      image = "projects/deeplearning-platform-release/global/images/family/common-cu128-ubuntu-2204-nvidia-570"
      size  = "150" # Increased to 150GB for multiple large AI models
    }
  }

  # Attach an NVIDIA T4 GPU
  guest_accelerator {
    type  = "nvidia-tesla-t4"
    count = 1
  }

  # Service account for GCP API access
  service_account {
    scopes = ["cloud-platform"]
  }

  # Network configuration with a public IP
  network_interface {
    network = "default"
    access_config {
      // Ephemeral public IP
    }
  }

  # Tags for applying firewall rules
  tags = ["http-server", "ssh-server"]

  # Ensures GPU drivers are handled correctly by the OS
  scheduling {
    on_host_maintenance = "TERMINATE"
  }

  # Execute our setup script on the first startup
  metadata_startup_script = file("startup-script.sh")
}

# Firewall rule to allow HTTP traffic on port 8000
resource "google_compute_firewall" "allow_http_8000" {
  name    = "allow-kalaasetu-http-8000"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["8000"] # Port for our API Gateway
  }

  target_tags   = ["http-server"]
  source_ranges = ["0.0.0.0/0"] # Allow traffic from any IP
}

# Firewall rule to allow SSH traffic on port 22
resource "google_compute_firewall" "allow_ssh" {
  name    = "allow-kalaasetu-ssh"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["22"] # Standard SSH port
  }

  target_tags   = ["ssh-server"]
  # WARNING: This allows SSH from any IP. For production, restrict this to your IP.
  source_ranges = ["0.0.0.0/0"] 
}

# Output the public IP address of the instance once created
output "instance_ip" {
  value = google_compute_instance.kalaasetu_vm.network_interface[0].access_config[0].nat_ip
}