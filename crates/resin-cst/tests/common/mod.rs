use resin_cst::Document;
use resin_executor::{Cancellation, Execution};

pub async fn parse(text: impl Into<String> + Send, previous: Option<&Document>) -> Document {
    resin_cst::build_cst(text, previous, &Execution::default(), &Cancellation::new())
        .await
        .expect("CST worker completed")
}
