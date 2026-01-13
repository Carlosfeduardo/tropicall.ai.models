"""
Tests for TextNormalizer.
"""

import pytest

from src.core.normalizer import TextNormalizer


class TestTextNormalizer:
    """Tests for TextNormalizer."""

    @pytest.fixture
    def normalizer(self):
        return TextNormalizer()

    def test_remove_urls(self, normalizer):
        """Should replace URLs with 'link'."""
        text = "Visite https://example.com para mais info"
        result = normalizer.normalize(text)
        assert "https://example.com" not in result
        assert "link" in result

    def test_expand_abbreviations(self, normalizer):
        """Should expand common abbreviations."""
        text = "Dr. João e Sra. Maria"
        result = normalizer.normalize(text)
        assert "Doutor João" in result
        assert "Senhora Maria" in result

    def test_normalize_dates(self, normalizer):
        """Should normalize dates to spoken form."""
        text = "Reunião em 15/03/2024"
        result = normalizer.normalize(text)
        assert "15 de março de 2024" in result

    def test_normalize_time(self, normalizer):
        """Should normalize time to spoken form."""
        text = "Encontro às 14:30"
        result = normalizer.normalize(text)
        assert "14 horas e 30 minutos" in result

    def test_normalize_time_full_hour(self, normalizer):
        """Should handle full hours."""
        text = "Reunião às 10:00"
        result = normalizer.normalize(text)
        assert "10 horas" in result
        assert "minutos" not in result.replace("10 horas", "")

    def test_normalize_currency(self, normalizer):
        """Should normalize currency to spoken form."""
        text = "Preço: R$ 1.500,00"
        result = normalizer.normalize(text)
        assert "reais" in result

    def test_normalize_percentages(self, normalizer):
        """Should normalize percentages."""
        text = "Aumento de 15%"
        result = normalizer.normalize(text)
        assert "15 por cento" in result

    def test_clean_whitespace(self, normalizer):
        """Should clean up extra whitespace."""
        text = "Texto   com    muitos   espaços"
        result = normalizer.normalize(text)
        assert "  " not in result

    def test_multiple_normalizations(self, normalizer):
        """Should handle multiple normalizations in one text."""
        text = "Dr. João visitou https://site.com em 25/12/2024 às 15:30 por R$ 100,00"
        result = normalizer.normalize(text)
        
        assert "Doutor João" in result
        assert "link" in result
        assert "25 de dezembro de 2024" in result
        assert "15 horas e 30 minutos" in result
        assert "reais" in result

    def test_preserve_normal_text(self, normalizer):
        """Should preserve text that doesn't need normalization."""
        text = "Este é um texto normal sem nada especial"
        result = normalizer.normalize(text)
        assert result == text
