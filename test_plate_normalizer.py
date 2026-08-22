# test_plate_normalizer.py
#
# اجرا با: pytest test_plate_normalizer.py -v
# یا اگه pytest نصب نیست: python test_plate_normalizer.py

import pytest
from plate_normalizer import normalize_plate, is_normalizable, InvalidPlateFormatError


# ─────────────── تست‌های ورودی معتبر ───────────────

def test_example_from_document():
    """همون مثالی که خود سند داده: ۵۹ ب ۶۹۴۶۷ باید بشه 59ب69467"""
    assert normalize_plate("۵۹ ب ۶۹۴۶۷") == "59ب69467"


def test_already_normalized_input():
    """اگه ورودی از قبل تمیز بود، نباید خراب بشه"""
    assert normalize_plate("59ب69467") == "59ب69467"


def test_persian_digits_converted():
    """ارقام فارسی باید به انگلیسی تبدیل بشن"""
    assert normalize_plate("۱۲ج۳۴۵۶۷") == "12ج34567"


def test_arabic_digits_converted():
    """ارقام عربی (٤ نه ۴) هم باید به انگلیسی تبدیل بشن"""
    assert normalize_plate("١٢ج٣٤٥٦٧") == "12ج34567"


def test_removes_spaces():
    """فاصله‌های اضافه بین بخش‌های پلاک باید حذف بشن"""
    assert normalize_plate("12 ج 34567") == "12ج34567"


def test_removes_dashes():
    """خط‌تیره‌های احتمالی بین بخش‌ها باید حذف بشن"""
    assert normalize_plate("12-ج-34567") == "12ج34567"


def test_removes_zero_width_non_joiner():
    """نیم‌فاصله (ZWNJ) باید حذف بشه"""
    assert normalize_plate("12\u200cج\u200c34567") == "12ج34567"


def test_arabic_yeh_converted_to_persian():
    """حرف ي عربی باید به ی فارسی تبدیل بشه"""
    # ي (arabic yeh, U+064A) -> ی (persian yeh, U+06CC)
    result = normalize_plate("12ي34567")
    assert result == "12ی34567"


def test_arabic_kaf_converted_to_persian():
    """حرف ك عربی باید به ک فارسی تبدیل بشه"""
    # ك (arabic kaf, U+0643) -> ک (persian kaf, U+06A9)
    result = normalize_plate("12ك34567")
    assert result == "12ک34567"


# ─────────────── تست‌های ورودی نامعتبر ───────────────

def test_empty_string_raises_error():
    with pytest.raises(InvalidPlateFormatError):
        normalize_plate("")


def test_none_raises_error():
    with pytest.raises(InvalidPlateFormatError):
        normalize_plate(None)


def test_too_short_raises_error():
    """کمتر از ۸ کاراکتر باید رد بشه"""
    with pytest.raises(InvalidPlateFormatError):
        normalize_plate("12ج345")


def test_too_long_raises_error():
    """بیشتر از ۸ کاراکتر باید رد بشه"""
    with pytest.raises(InvalidPlateFormatError):
        normalize_plate("12ج3456789")


def test_missing_letter_raises_error():
    """اگه جای حرف هم رقم باشه (یعنی ۸ رقم پشت سر هم)، باید رد بشه"""
    with pytest.raises(InvalidPlateFormatError):
        normalize_plate("12345678")


def test_letter_in_wrong_position_raises_error():
    """اگه حرف جای رقم دوم باشه (نه جایگاه سوم)، باید رد بشه"""
    with pytest.raises(InvalidPlateFormatError):
        normalize_plate("1جع34567")


def test_invalid_letter_raises_error():
    """حرفی که جزو حروف مجاز پلاک ایرانی نیست باید رد بشه"""
    with pytest.raises(InvalidPlateFormatError):
        normalize_plate("12z34567")  # z جزو حروف پلاک ایرانی نیست


def test_only_whitespace_raises_error():
    """رشته‌ای که فقط فاصله و نیم‌فاصله‌ست باید رد بشه"""
    with pytest.raises(InvalidPlateFormatError):
        normalize_plate("   \u200c  ")


# ─────────────── تست‌های تابع کمکی is_normalizable ───────────────

def test_is_normalizable_true_for_valid():
    assert is_normalizable("۵۹ ب ۶۹۴۶۷") is True


def test_is_normalizable_false_for_invalid():
    assert is_normalizable("چیز نامعتبر") is False


# اگه pytest نصب نبود، این بخش امکان اجرای دستی رو می‌ده
if __name__ == "__main__":
    import sys
    print("در حال اجرای تست‌ها بدون pytest (حالت ساده)...")
    test_functions = [obj for name, obj in list(globals().items())
                       if name.startswith("test_") and callable(obj)]
    passed, failed = 0, 0
    for fn in test_functions:
        try:
            fn()
            passed += 1
            print(f"  ✓ {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  ✗ {fn.__name__} -> {e}")
    print(f"\nنتیجه: {passed} موفق، {failed} ناموفق")
    sys.exit(1 if failed else 0)
