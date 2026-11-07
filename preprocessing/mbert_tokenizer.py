
from transformers import AutoTokenizer

MODEL_NAME = "bert-base-multilingual-cased"


def load_tokenizer():

    print("ĐANG TẢI TOKENIZER mBERT...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("TẢI TOKENIZER THÀNH CÔNG")
    print("Model:", MODEL_NAME)
    print("Vocabulary size:", tokenizer.vocab_size)

    return tokenizer


if __name__ == "__main__":

    tokenizer = load_tokenizer()

    sample = (
        "Ma OTP xac thuc GD la 066595, "
        "hieu luc [TIME]."
    )

    encoded = tokenizer(
        sample,
        add_special_tokens=True
    )

    print("\nVĂN BẢN GỐC:")
    print(sample)

    print("\nTOKENS:")
    print(tokenizer.convert_ids_to_tokens(
        encoded["input_ids"]
    ))

    print("\nINPUT IDS:")
    print(encoded["input_ids"])
