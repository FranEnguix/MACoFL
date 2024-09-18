import copy

from aioxmpp import JID
from spade.message import Message


class MultipartHandler:
    """
    Class created to handle the SPADE agents maximum message length limitation. The aioxmpp package maximum
    content length is 256 * 1024, so this class is able to split a content that exceeds a desired size into messages
    of the same desired maximum length. This class inyects a header into the messages content to be able to
    rebuild the messages in the correct order. The header is "multipart#[index]/[total]|" where "index"
    is the id of the current message, starting by 1, and "total" is the number of messages needed to rebuild the
    original content.
    """

    def __init__(self) -> None:
        self.multipart_message_storage: dict[JID, str] = {}
        self.metadata_split: str = "|"
        self.metadata_header_size: int = len(
            f"multipart#9999/9999{self.metadata_split}"
        )

    def is_multipart(self, message: Message) -> bool:
        return message.body.startswith("multipart#")

    def any_multipart_waiting(self) -> bool:
        return len(self.multipart_message_storage.keys()) > 0

    def is_multipart_complete(self, message: Message) -> bool | None:
        """
        Returns a bool to denote whether the message is complete and ready to be rebuilded.
        Returns None if the sender has not multipart messages stored.

        Args:
            message (Message): A SPADE message used to check if it is part of a chain of multipart messages.

        Returns:
            bool | None: True if multipart is complete, False otherwise and None if the sender has not multipart messages stored.
        """
        sender = message.sender
        if not sender in self.multipart_message_storage.keys():
            return None
        for part in self.multipart_message_storage[sender]:
            if part is None:
                return False
        return True

    def rebuild_multipart_content(self, sender: JID) -> str:
        content = ""
        for part in self.multipart_message_storage[sender]:
            content += part
        return content

    def remove_data(self, sender: JID) -> None:
        if sender in self.multipart_message_storage.keys():
            del self.multipart_message_storage[sender]

    def rebuild_multipart(self, message: Message) -> Message | None:
        """
        Rebuilds the multipart message linked to the message argument and removes the sender
        multipart stored data.

        Args:
            message (Message): One message part.

        Returns:
            Message | None: The rebuilded message with all the multiparts content in its body property.
            Returns None if the message is not complete or it is not a multipart message.
        """
        # NOTE multipart header: multipart#1/2|
        # multipart_meta = message.get_metadata("multipart")
        if self.is_multipart(message):
            sender = message.sender
            multipart_meta = message.body.split(self.metadata_split)[0]
            multipart_meta_parts = multipart_meta.split("#")[1]
            part_number = int(multipart_meta_parts.split("/")[0])
            total_parts = int(multipart_meta_parts.split("/")[1])
            if not sender in self.multipart_message_storage.keys():
                self.multipart_message_storage[sender] = [None] * total_parts
            self.multipart_message_storage[sender][part_number - 1] = message.body[
                len(multipart_meta + self.metadata_split) :
            ]
            if self.is_multipart_complete(message):
                message.body = self.rebuild_multipart_content(sender=sender)
                self.remove_data(sender=sender)
                return message
        return None

    def divide_content(self, content: str, size: int) -> list[str]:
        return [content[i : i + size] for i in range(0, len(content), size)]

    def generate_multipart_content(
        self, content: str, max_size: int
    ) -> list[str] | None:
        """
        Generates a list of multipart content based on the desired maximum size of each multipart message content
        and the maximum header size of the multipart messages metadata.

        Args:
            content (str): The content to be splitted if its length exceeds the max_size.
            max_size (int): Threshold to split the content into a list of content.

        Returns:
            list[str] | None: List of multipart message content to put in the body of the SPADE messages
            or None if the content length does not exceed the max_size tanking into account the multipart header metadata.
        """
        if len(content) > max_size:
            multiparts = self.divide_content(
                content, max_size - self.metadata_header_size
            )
            return [
                f"multipart#{i + 1}/{len(multiparts)}{self.metadata_split}{part}"
                for i, part in enumerate(multiparts)
            ]
        return None

    def generate_multipart_messages(
        self, content: str, max_size: int, message_base: Message
    ) -> list[Message] | None:
        """
        Creates multipart messages from one SPADE message if the length of the content
        argument is longer than the max_size argument.

        Args:
            content (str): The information that multipart messages will have in its bodies.
            max_size (int): Maximum size body length of each multipart message.
            message_base (Message): The message that will be copied by all the multipart messages, replacing just the body.

        Returns:
            list[Message] | None: A list of multipart messages to send or None if the content does not exceed the maximum size.
        """
        multiparts = self.generate_multipart_content(content=content, max_size=max_size)
        if multiparts is not None:
            multiparts_messages = []
            for multipart in multiparts:
                message = copy.deepcopy(message_base)
                message.body = multipart
                multiparts_messages.append(message)
            return multiparts_messages
        return None
