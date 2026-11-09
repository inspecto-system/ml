import pytest

from tests.audit.helpers import ImageFactory, make_picture


@pytest.fixture
def picture() -> ImageFactory:
    return make_picture
