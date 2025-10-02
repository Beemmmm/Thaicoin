# To learn more about how to use Nix to configure your environment
# see: https://firebase.google.com/docs/studio/customize-workspace
{ pkgs, ... }: {
  # Which nixpkgs channel to use.
  channel = "stable-24.05"; # or "unstable"

  # Use https://search.nixos.org/packages to find packages
  packages = [
    # 1. ติดตั้ง Python 3.11 เวอร์ชันหลัก
    pkgs.python311

    # 2. ติดตั้ง Dependencies ของ Python ที่จำเป็น **สำหรับ Flask เท่านั้น**
    #    เราต้องคอมเมนต์ไลบรารีขนาดใหญ่ออกไปก่อน เพื่อให้ Environment สร้างสำเร็จ
    pkgs.python311Packages.flask
    pkgs.python311Packages.requests # (ถ้าใช้ในการเรียก API)
    
    # pkgs.python311Packages.numpy    # <-- คอมเมนต์ไว้ก่อน
    # pkgs.python311Packages.opencv-python # <-- คอมเมนต์ไว้ก่อน
    # pkgs.python311Packages.torch   # <-- คอมเมนต์ไว้ก่อน
  ];

  # Sets environment variables in the workspace
  env = {};
  idx = {
    # Search for the extensions you want on https://open-vsx.org/ and use "publisher.id"
    extensions = [
      # "vscodevim.vim"
    ];

    # Enable previews
    previews = {
      enable = true;
      previews = {
        # รันไฟล์ app.py และใช้พอร์ตที่ Firebase Studio กำหนด
        web = {
          command = ["python" "app.py"];
          manager = "web";
          env = {
            PORT = "$PORT";
          };
        };
      };
    };

    # Workspace lifecycle hooks
    workspace = {
      # Runs when a workspace is first created
      onCreate = {};
      # Runs when the workspace is (re)started
      onStart = {};
    };
  };
}