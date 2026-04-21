"""Tests for the regex-based contact extractor."""

import pytest

from hansel.cv.regex_parser import extract_contact


class TestEmailExtraction:
    
    def test_standard_email(self):
        result = extract_contact("Contact: maria.lopez@example.com")
        assert result.email == "maria.lopez@example.com"
    
    def test_email_with_plus_and_dots(self):
        result = extract_contact("Reach me at: user.name+tag@domain.co.uk")
        assert result.email == "user.name+tag@domain.co.uk"
    
    def test_no_email(self):
        result = extract_contact("No email here")
        assert result.email is None
    
    def test_picks_first_email(self):
        """If multiple emails, we take the first one."""
        result = extract_contact("a@b.com and c@d.com")
        assert result.email == "a@b.com"


class TestPhoneExtraction:
    
    def test_international_with_spaces(self):
        result = extract_contact("Tel: +34 657 641 482")
        assert result.phone is not None
        assert "657" in result.phone
    
    def test_without_country_code(self):
        result = extract_contact("Call 612 345 678")
        assert result.phone is not None
        assert "612" in result.phone
    
    def test_no_phone(self):
        result = extract_contact("No phone number here")
        assert result.phone is None


class TestLinkedInExtraction:
    
    def test_url_format(self):
        result = extract_contact("Profile: https://linkedin.com/in/maria-lopez")
        assert result.linkedin is not None
        assert "maria-lopez" in result.linkedin
    
    def test_label_format(self):
        """Plain 'LinkedIn: Name' format from many Spanish CVs."""
        text = "LinkedIn: Maria Lopez Garcia | Available now"
        result = extract_contact(text)
        assert result.linkedin is not None
        # Should strip 'LinkedIn:' prefix and stop at pipe
        assert "LinkedIn:" not in result.linkedin
    
    def test_no_linkedin(self):
        result = extract_contact("No social profiles")
        assert result.linkedin is None


class TestGitHubExtraction:
    
    def test_standard_url(self):
        result = extract_contact("Code: github.com/marialopez")
        assert result.linkedin != result.github  # just sanity
        assert result.github == "github.com/marialopez"
    
    def test_case_insensitive(self):
        result = extract_contact("Repo: GitHub.com/Maria")
        assert result.github is not None
        assert "Maria" in result.github


class TestFullExtraction:
    
    def test_complete_cv_header(self):
        """All fields present in a realistic CV header."""
        text = """
        MARIA LOPEZ GARCIA
        Junior Backend Developer
        
        maria@example.com | +34 612 345 678 | Valencia
        LinkedIn: maria-lopez-dev | github.com/maria-lopez
        """
        result = extract_contact(text)
        assert result.email == "maria@example.com"
        assert result.phone is not None
        assert result.linkedin is not None
        assert result.github == "github.com/maria-lopez"
    
    def test_empty_input(self):
        result = extract_contact("")
        assert result.email is None
        assert result.phone is None
        assert result.linkedin is None
        assert result.github is None