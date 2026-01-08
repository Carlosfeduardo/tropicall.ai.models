"""Text normalizer for Brazilian Portuguese TTS."""

import re


class TextNormalizer:
    """
    Normalizes text for better TTS quality in pt-BR.
    
    Handles:
    - URLs (replaced with "link")
    - Common abbreviations
    - Dates
    - Numbers (basic)
    """

    # URL pattern
    URL_PATTERN = re.compile(r"https?://\S+")
    
    # Date pattern (DD/MM/YYYY)
    DATE_PATTERN = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")
    
    # Time pattern (HH:MM)
    TIME_PATTERN = re.compile(r"(\d{1,2}):(\d{2})")
    
    # Currency pattern (R$ X.XXX,XX)
    CURRENCY_PATTERN = re.compile(r"R\$\s*(\d+(?:\.\d{3})*(?:,\d{2})?)")
    
    # Percentage pattern
    PERCENTAGE_PATTERN = re.compile(r"(\d+(?:,\d+)?)\s*%")
    
    # Common abbreviations (only with punctuation to avoid replacing parts of words)
    ABBREVIATIONS = {
        "Dr.": "Doutor",
        "Dra.": "Doutora",
        "Sr.": "Senhor",
        "Sra.": "Senhora",
        "Srta.": "Senhorita",
        "Prof.": "Professor",
        "Profa.": "Professora",
        "etc.": "etcetera",
        "ex.": "exemplo",
        "pág.": "página",
        "págs.": "páginas",
        "tel.": "telefone",
        "cel.": "celular",
        "nº": "número",
        "n°": "número",
        "Av.": "Avenida",
        "Ltda.": "Limitada",
        "S.A.": "Sociedade Anônima",
    }
    
    # Unit patterns - only match when preceded by number and followed by space/punctuation
    UNIT_PATTERNS = [
        (re.compile(r"(\d+)\s*km\b"), r"\1 quilômetros"),
        (re.compile(r"(\d+)\s*kg\b"), r"\1 quilogramas"),
        (re.compile(r"(\d+)\s*g\b"), r"\1 gramas"),
        (re.compile(r"(\d+)\s*m\b"), r"\1 metros"),
        (re.compile(r"(\d+)\s*cm\b"), r"\1 centímetros"),
        (re.compile(r"(\d+)\s*mm\b"), r"\1 milímetros"),
        (re.compile(r"(\d+)\s*h\b"), r"\1 horas"),
        (re.compile(r"(\d+)\s*min\b"), r"\1 minutos"),
        (re.compile(r"(\d+)\s*seg\b"), r"\1 segundos"),
    ]
    
    # Month names for date normalization
    MONTHS = {
        "01": "janeiro",
        "02": "fevereiro",
        "03": "março",
        "04": "abril",
        "05": "maio",
        "06": "junho",
        "07": "julho",
        "08": "agosto",
        "09": "setembro",
        "10": "outubro",
        "11": "novembro",
        "12": "dezembro",
    }

    def normalize(self, text: str) -> str:
        """
        Normalize text for TTS.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Remove URLs
        text = self.URL_PATTERN.sub("link", text)
        
        # Expand abbreviations (safe ones with punctuation)
        for abbr, full in self.ABBREVIATIONS.items():
            text = text.replace(abbr, full)
        
        # Expand unit abbreviations (only when preceded by numbers)
        for pattern, replacement in self.UNIT_PATTERNS:
            text = pattern.sub(replacement, text)
        
        # Normalize dates (DD/MM/YYYY -> DD de mês de YYYY)
        text = self._normalize_dates(text)
        
        # Normalize time (HH:MM -> HH horas e MM minutos)
        text = self._normalize_time(text)
        
        # Normalize currency
        text = self._normalize_currency(text)
        
        # Normalize percentages
        text = self._normalize_percentages(text)
        
        # Clean up extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        return text

    def _normalize_dates(self, text: str) -> str:
        """Normalize dates to spoken form."""
        def replace_date(match: re.Match) -> str:
            day = match.group(1).lstrip("0") or "0"
            month = match.group(2).zfill(2)
            year = match.group(3)
            month_name = self.MONTHS.get(month, month)
            return f"{day} de {month_name} de {year}"
        
        return self.DATE_PATTERN.sub(replace_date, text)

    def _normalize_time(self, text: str) -> str:
        """Normalize time to spoken form."""
        def replace_time(match: re.Match) -> str:
            hours = match.group(1).lstrip("0") or "0"
            minutes = match.group(2)
            
            if minutes == "00":
                return f"{hours} horas"
            else:
                minutes = minutes.lstrip("0") or "0"
                return f"{hours} horas e {minutes} minutos"
        
        return self.TIME_PATTERN.sub(replace_time, text)

    def _normalize_currency(self, text: str) -> str:
        """Normalize currency to spoken form."""
        def replace_currency(match: re.Match) -> str:
            value = match.group(1)
            # Remove thousands separator and convert decimal
            value = value.replace(".", "").replace(",", " vírgula ")
            return f"{value} reais"
        
        return self.CURRENCY_PATTERN.sub(replace_currency, text)

    def _normalize_percentages(self, text: str) -> str:
        """Normalize percentages to spoken form."""
        def replace_percentage(match: re.Match) -> str:
            value = match.group(1).replace(",", " vírgula ")
            return f"{value} por cento"
        
        return self.PERCENTAGE_PATTERN.sub(replace_percentage, text)


# Default instance
normalizer = TextNormalizer()
