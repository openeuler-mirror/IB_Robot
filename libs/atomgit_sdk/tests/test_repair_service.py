#!/usr/bin/env python3
"""
Test RepairService reply_to_comment functionality
"""

import pytest
from unittest.mock import Mock

from atomgit_sdk.services import RepairService


class TestRepairService:
    """Test suite for RepairService"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.mock_client = Mock()
        self.mock_client.config = Mock()
        self.mock_client.config.owner = "test_owner"
        self.mock_client.config.repo = "test_repo"
        self.service = RepairService(self.mock_client)
    
    def test_reply_to_comment_with_discussion_id(self):
        """Test reply with discussion_id"""
        # Mock get_pr_comments
        original_comment = {
            "id": 123,
            "discussion_id": "abc456",
            "body": "Original comment"
        }
        self.mock_client.get_pr_comments.return_value = [original_comment]
        
        # Mock request
        expected_response = {"id": 456, "body": "Reply"}
        self.mock_client.request.return_value = expected_response
        
        result = self.service.reply_to_comment(1, 123, "Test reply")
        assert result == expected_response
        
        # Verify correct API call
        self.mock_client.request.assert_called_once()
        args, kwargs = self.mock_client.request.call_args
        assert args == (
            "POST",
            "/api/v5/repos/test_owner/test_repo/pulls/1/comments",
        )
        assert kwargs["body"]["in_reply_to"] == 123
        assert kwargs["body"]["discussion_id"] == "abc456"
        assert "Test reply" in kwargs["body"]["body"]
    
    def test_reply_to_comment_without_discussion_id(self):
        """Test reply without discussion_id"""
        original_comment = {
            "id": 124,
            "body": "Original comment"
        }
        self.mock_client.get_pr_comments.return_value = [original_comment]
        self.mock_client.request.return_value = {"id": 789}
        
        result = self.service.reply_to_comment(1, 124, "Test reply")
        assert result == {"id": 789}
        
        # Verify in_reply_to is set
        self.mock_client.request.assert_called_once()
        args, kwargs = self.mock_client.request.call_args
        assert args == (
            "POST",
            "/api/v5/repos/test_owner/test_repo/pulls/1/comments",
        )
        assert kwargs["body"]["in_reply_to"] == 124
        assert "discussion_id" not in kwargs["body"]
    
    def test_reply_to_nonexistent_comment(self):
        """Test reply to non-existent comment"""
        self.mock_client.get_pr_comments.return_value = []
        
        with pytest.raises(ValueError, match="Comment 999 not found"):
            self.service.reply_to_comment(1, 999, "Test reply")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
