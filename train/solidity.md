# Solidity — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder sobre Solidity (smart contracts/blockchain).
**Expert sugerido**: `blockchain_experts`. **Total est.**: ~85 lições.
**Convenção**: `treinamento_solidity/<família>/<subsetor>/*.md` → path = [família, subsetor].

## fundamentos/ — ~15
o que é blockchain e Ethereum; EVM e gas; o que é um smart contract; estrutura de um contrato; versão do compilador (`pragma`); tipos de dados; variáveis de estado vs locais; visibilidade (public/private/internal/external); funções; constructor; comentários e NatSpec; deploy (visão geral); Remix IDE; contas e endereços.

## linguagem/ — ~25
tipos (uint/int/address/bool/bytes); strings; arrays (fixos e dinâmicos); mappings; structs; enums; controle de fluxo; loops; modifiers; eventos e logs; `require`/`revert`/`assert`; error handling; custom errors; herança; interfaces; abstract contracts; libraries; `payable` e envio de ether; `msg.sender`/`msg.value`; memory vs storage vs calldata; funções view/pure; fallback e receive.

## seguranca-padroes/ — ~25
padrões de segurança; reentrancy e proteção; overflow/underflow (SafeMath); checks-effects-interactions; access control (Ownable); tokens ERC-20; tokens ERC-721 (NFT); ERC-1155; OpenZeppelin; upgradeable contracts (proxy); DeFi básico; oracles (Chainlink); assinaturas e ecrecover; gas optimization; ataques comuns; auditoria; testes de segurança.

## ecossistema/ — ~20
Hardhat; Foundry; testes (Mocha/Chai/Forge); ethers.js/web3.js; deploy scripts; redes de teste (testnets); MetaMask; interação front-end (dApps); The Graph; IPFS; verificação de contratos (Etherscan); Layer 2; gas reporting; forking; boas práticas; comparação com outras chains (Solana/Vyper).
