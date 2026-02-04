if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
    exit 0
fi

cargo asm --release --lib --asm $1
