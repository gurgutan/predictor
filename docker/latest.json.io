{
    "Id": "sha256:7ca67d5dbe5ded202bd51ed2f39aa0e4b211d23470348768d68be10f312ef92c",
    "RepoTags": [
        "intel/intel-optimized-tensorflow:latest",
        "intelaipg/intel-optimized-tensorflow:latest"
    ],
    "RepoDigests": [
        "intel/intel-optimized-tensorflow@sha256:2bf3f5f22f1dd81c7aeccddc00857d229e740e9e68680aaf0453cca53532d888",
        "intelaipg/intel-optimized-tensorflow@sha256:2bf3f5f22f1dd81c7aeccddc00857d229e740e9e68680aaf0453cca53532d888"
    ],
    "Parent": "",
    "Comment": "",
    "Created": "2020-08-30T19:21:12.95307351Z",
    "Container": "756633be09859a648891e502cfaf9225ef017e9b7ace34a170bbeed9910a76a9",
    "ContainerConfig": {
        "Hostname": "756633be0985",
        "Domainname": "",
        "User": "",
        "AttachStdin": true,
        "AttachStdout": true,
        "AttachStderr": true,
        "Tty": true,
        "OpenStdin": true,
        "StdinOnce": true,
        "Env": [
            "http_proxy=http://proxy-us.intel.com:911",
            "https_proxy=http://proxy-us.intel.com:911",
            "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "LANG=C.UTF-8"
        ],
        "Cmd": [
            "/bin/bash"
        ],
        "Image": "52f9a17131fd",
        "Volumes": null,
        "WorkingDir": "",
        "Entrypoint": null,
        "OnBuild": null,
        "Labels": {
            "maintainer": "Clayne Robison <clayne.b.robison@intel.com>"
        }
    },
    "DockerVersion": "19.03.8",
    "Author": "",
    "Config": {
        "Hostname": "756633be0985",
        "Domainname": "",
        "User": "",
        "AttachStdin": true,
        "AttachStdout": true,
        "AttachStderr": true,
        "Tty": true,
        "OpenStdin": true,
        "StdinOnce": true,
        "Env": [
            "http_proxy=http://proxy-us.intel.com:911",
            "https_proxy=http://proxy-us.intel.com:911",
            "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "LANG=C.UTF-8"
        ],
        "Cmd": [
            "/bin/bash"
        ],
        "Image": "52f9a17131fd",
        "Volumes": null,
        "WorkingDir": "",
        "Entrypoint": null,
        "OnBuild": null,
        "Labels": {
            "maintainer": "Clayne Robison <clayne.b.robison@intel.com>"
        }
    },
    "Architecture": "amd64",
    "Os": "linux",
    "Size": 3840081208,
    "VirtualSize": 3840081208,
    "GraphDriver": {
        "Data": {
            "LowerDir": "/var/lib/docker/overlay2/d4de6d583a998ea6c72b5bab95d52170dd8d0d5cbc43cd15d7330b413535fe93/diff:/var/lib/docker/overlay2/a9ff0c78c6f7fcd4251d0f592e1558e7c08b2c20245ba2dafb19c75348397ceb/diff:/var/lib/docker/overlay2/dbb35e6b882053396ce3b5bfce833ee017d2a62b8ebeb54245dea23f28ecbb1e/diff:/var/lib/docker/overlay2/7861635b52845ed0938b773eca26164a6a6aaedeb590221cfb9e4615729a30a0/diff:/var/lib/docker/overlay2/ae6f94c582a487b3f2a28d4f01b05d263e363d5d4d5b311ac0f66838cf2a5eff/diff:/var/lib/docker/overlay2/a01eb812c269af8e224697df9dcb15c713e9c4d68b1e65d32d58a8b46ad2505f/diff:/var/lib/docker/overlay2/43fa3a50194b0a4b8b41510302db6375d349a1679746f7d7fb10ae37bb637212/diff:/var/lib/docker/overlay2/1f1db46178b849ad19dfde2c00139a489eccc8a05fcf8aa1d64896e9daf78681/diff:/var/lib/docker/overlay2/1c6a77883921e8cee044a93ace3d073ba562d4033efee37e28163cbe96ab9422/diff:/var/lib/docker/overlay2/29db36b3b41c3d713a861c47cdd74c40d157a23a4773d4b1f6f3d318413a0784/diff:/var/lib/docker/overlay2/096cb316d751be42a1de7970d8dbffa54cfe7058c6ccb271cb85bc8abf85678c/diff:/var/lib/docker/overlay2/a8629473d73c365e00f4e54817068ae99b9a86d57080ca2783d9ece6270a233a/diff:/var/lib/docker/overlay2/16947600c8b41449560fe7456e0252f7438bbf23f0b702cf2b911ac3327bc315/diff:/var/lib/docker/overlay2/5152125db62b5b132438e3c9a2f7b1a6f7ce99239fcc6748ab86e22ac9b3ed5f/diff:/var/lib/docker/overlay2/ee61a2fa9d290ffe8ba96b835a640a84126b645e180be57e3856ac2851fa8cb6/diff",
            "MergedDir": "/var/lib/docker/overlay2/b57a7702869de9c7ffb5836aaa03943d32e04c2e20508f635b2b578440f8f39c/merged",
            "UpperDir": "/var/lib/docker/overlay2/b57a7702869de9c7ffb5836aaa03943d32e04c2e20508f635b2b578440f8f39c/diff",
            "WorkDir": "/var/lib/docker/overlay2/b57a7702869de9c7ffb5836aaa03943d32e04c2e20508f635b2b578440f8f39c/work"
        },
        "Name": "overlay2"
    },
    "RootFS": {
        "Type": "layers",
        "Layers": [
            "sha256:7ef3687765828a9cb2645925f27febbac21a5adece69e8437c26184a897b6ec7",
            "sha256:83f4287e1f0496940f8c222ca09cbaf2c7f564a10c57b4609800babe8d1b5b32",
            "sha256:d3a6da143c913c5e605737a9d974638f75451b5c593e58eb7e132fcf0e23c6db",
            "sha256:8682f9a74649fb9fc5d14f827a35259aae8b58c57df8d369f6aa2e92865930c2",
            "sha256:04ec0c2fd468f144c8c258edda1bf1b21d044e7206c1667823972879c38e1783",
            "sha256:a7bb721015400d414e5e78ba9223203b756262060e2a719eef78901f3e6cbe19",
            "sha256:2cdb2c6a3a9ced4c54a368a32dc3eac08a42f5ebcb7f1d8544651d7835d12b34",
            "sha256:8001ac9daff0513dee87c77c58c76b03cc9e43be553d232b080f194ef6d40eb2",
            "sha256:8e1d800aec8e0c7d5d8c812a36690fda3a825f6f3a46926ab0c1847735c14306",
            "sha256:cbad238d35116b5f1aa0a4a770f9feefe27da3b30273e053838097b374d1f62b",
            "sha256:15d8713e46ff6fc5447811580cbda7dc16a3eadd2502ad4f42f017d30c6f5174",
            "sha256:e566189fc15bddab72f674a1d58db5957aff90412ffc0f25d1001da6e4a8977a",
            "sha256:23504765f79c79e1c6907964f07984a2c5aa4815f4bf1b2c869d9d5f23ec3377",
            "sha256:10942266a4a12036c5194aa6a116139af7c6171c1e358b2119bb4abac548406f",
            "sha256:5fe13870977d9d92b9f1c9549e98d91b04326cbae308dcf01133517dbbb8cfbc",
            "sha256:4c9564408c3638d690e2b62c8f3012648863e2059bb4706bab9678026c62857e"
        ]
    },
    "Metadata": {
        "LastTagTime": "0001-01-01T00:00:00Z"
    },
    "CreatedTime": 1598815272953
}