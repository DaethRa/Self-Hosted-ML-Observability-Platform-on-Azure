terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "mlops_rg" {
  name     = "olist-mlops-rg-austria-main"
  location = "Austria East"
}

resource "azurerm_virtual_network" "vnet" {
  name                = "mlops-vnet"
  address_space       = ["10.0.0.0/16"]
  location            = azurerm_resource_group.mlops_rg.location
  resource_group_name = azurerm_resource_group.mlops_rg.name
}

resource "azurerm_subnet" "subnet" {
  name                 = "mlops-subnet"
  resource_group_name  = azurerm_resource_group.mlops_rg.name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = ["10.0.1.0/24"]
}

resource "azurerm_public_ip" "public_ip" {
  name                = "mlops-public-ip"
  location            = azurerm_resource_group.mlops_rg.location
  resource_group_name = azurerm_resource_group.mlops_rg.name
  allocation_method   = "Static"
  sku                 = "Standard"
}

resource "azurerm_network_security_group" "nsg" {
  name                = "mlops-nsg"
  location            = azurerm_resource_group.mlops_rg.location
  resource_group_name = azurerm_resource_group.mlops_rg.name

  security_rule {
    name                       = "SSH"
    priority                   = 1001
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "22"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
}

resource "azurerm_network_interface" "nic" {
  name                = "mlops-nic"
  location            = azurerm_resource_group.mlops_rg.location
  resource_group_name = azurerm_resource_group.mlops_rg.name

  ip_configuration {
    name                          = "internal"
    subnet_id                     = azurerm_subnet.subnet.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.public_ip.id
  }
}

resource "azurerm_network_interface_security_group_association" "nic_nsg" {
  network_interface_id      = azurerm_network_interface.nic.id
  network_security_group_id = azurerm_network_security_group.nsg.id
}

resource "azurerm_linux_virtual_machine" "vm" {
  name                = "mlops-vm-main"
  resource_group_name = azurerm_resource_group.mlops_rg.name
  location            = azurerm_resource_group.mlops_rg.location
  size                = "Standard_B2s_v2"
  admin_username      = "azureuser"
  network_interface_ids = [azurerm_network_interface.nic.id]

  admin_ssh_key {
    username   = "azureuser"
    public_key = file("~/.ssh/id_rsa.pub")
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Standard_LRS"
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "0001-com-ubuntu-server-jammy"
    sku       = "22_04-lts"
    version   = "latest"
  }

  custom_data = filebase64("${path.module}/cloud-init.yaml")
}

resource "random_id" "storage_id" {
  byte_length = 4
}

resource "azurerm_storage_account" "mlflow_storage" {
  name                     = "mlflowblob${random_id.storage_id.hex}"
  resource_group_name      = azurerm_resource_group.mlops_rg.name
  location                 = azurerm_resource_group.mlops_rg.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
}

output "public_ip_address" {
  value = azurerm_public_ip.public_ip.ip_address
}

resource "random_string" "acr_name" {
  length  = 8
  upper   = false
  special = false
}

resource "azurerm_container_registry" "acr" {
  name                = "olistmlopsacr${random_string.acr_name.result}"
  resource_group_name = azurerm_resource_group.mlops_rg.name
  location            = azurerm_resource_group.mlops_rg.location
  sku                 = "Basic"
  admin_enabled       = true
}

output "acr_login_server" {
  value = azurerm_container_registry.acr.login_server
}

output "acr_username" {
  value = azurerm_container_registry.acr.admin_username
  sensitive = true
}

output "acr_password" {
  value = azurerm_container_registry.acr.admin_password
  sensitive = true
}

resource "azurerm_storage_container" "mlflow_container" {
  name                  = "mlflow-artifacts"
  storage_account_name  = azurerm_storage_account.mlflow_storage.name
  container_access_type = "private"
}

output "storage_account_name" {
  value = azurerm_storage_account.mlflow_storage.name
}

output "storage_connection_string" {
  value     = azurerm_storage_account.mlflow_storage.primary_connection_string
  sensitive = true
}