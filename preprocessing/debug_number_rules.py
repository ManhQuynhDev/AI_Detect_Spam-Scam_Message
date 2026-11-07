
from pathlib import Path
import pandas as pd
import re

BASE_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE_DIR / "output" / "cleaned_dataset.csv")

TARGET_IDS = ["COM_851", "COM_863", "COM_884"]

for _, row in df[df["message_id"].isin(TARGET_IDS)].iterrows():
    text = str(row["text_clean"])

    print("\n" + "=" * 80)
    print("MESSAGE:", row["message_id"])
    print("TEXT:", text)

    for match in re.finditer(r"\[NUMBER\]", text, re.IGNORECASE):
        before = text[max(0, match.start() - 65):match.start()]
        after = text[match.end():match.end() + 65]

        print("\nOccurrence:", text[:match.start()].count("[NUMBER]") + 1)
        print("BEFORE:", repr(before))
        print("AFTER:", repr(after))

        print(
            "YEAR MATCH:",
            bool(re.search(r"\bnăm\s*$", before.lower()))
        )

        print(
            "OTP MATCH:",
            bool(re.search(
                r"(?:là|la)\s+mã\s+otp",
                after.lower()
            ))
        )
