use std::fmt;

use crate::Span;

pub type Result<T> = std::result::Result<T, Outbox>;

#[derive(Debug)]
pub struct Outbox {
    pub messages: Vec<Message>,
}

impl fmt::Display for Outbox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for msg in &self.messages {
            if let Some(span) = &msg.span {
                writeln!(f, "{} at {}", msg.title, span)?;
            } else {
                writeln!(f, "{}", msg.title)?;
            }
            for note in &msg.notes {
                if let Some(span) = &note.span {
                    writeln!(f, "  note: {} at {}", note.caption, span)?;
                } else {
                    writeln!(f, "  note: {}", note.caption)?;
                }
            }
        }
        Ok(())
    }
}
#[derive(Debug)]
pub struct Message {
    pub title: String,
    pub span: Option<Span>,
    pub notes: Vec<Note>,
}
#[derive(Debug)]
pub struct Note {
    pub caption: String,
    pub span: Option<Span>,
}
