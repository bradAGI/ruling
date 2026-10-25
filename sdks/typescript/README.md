# ruling TypeScript client

Zero-dependency ESM client for a running ruling server. Node 18+ or any browser.

```js
import { RulingClient, choice, noul, score } from "ruling";

const client = new RulingClient({ baseUrl: "http://127.0.0.1:8010" });
const { answers } = await client.systemOne({
  state: { message: "My card was charged twice. Refund the duplicate." },
  questions: {
    team: choice("Which team should handle this?", { billing: "Charges and refunds", technical: "Bugs" }),
    urgency: score("How urgent is this?", ["Can wait", "This week", "Today"]),
    refund: noul("The customer asks for a refund."),
  },
});
if (answers.team.confidence > 0.8 && answers.refund.noul > 0.9) routeToBilling();
```

Pass `apiKey` when the server runs with `RULING_API_KEY`. Errors throw `RulingError`
with the HTTP status and the server's detail.

Images go in the state as `image(bytes, "image/png")` blocks, or beside it as
`assets: { "<photo>": image(...) }` named wherever the text says `<photo>`; a
`box("the total line on <image 1>")` question answers with the edges of what it
finds, as fractions of the image. `longContext: true` asks for the server's
long-context limits.

When the server keeps a ledger (`RULING_LEDGER`), each response has an `id`;
`client.outcome(id, { team: "billing" })` reports what turned out to be right,
and `systemOne({ ..., excludeFromTraining: true })` keeps a decision out of it.
`client.outcome(id, undefined, { team: { correct: true } })` marks a served
answer right or wrong instead, and `client.outcomes([{ id, labels }, …])`
reports up to 1,000 at once, stored all or none.
